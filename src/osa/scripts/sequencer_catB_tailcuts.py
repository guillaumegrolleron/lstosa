import glob
import re
import shlex
import argparse
import logging
from pathlib import Path
from astropy.table import Table
import subprocess as sp

from osa.configs import options
from osa.configs.config import cfg
from osa.nightsummary.extract import get_last_pedcalib
from osa.utils.cliopts import valid_date, set_default_date_if_needed
from osa.utils.logging import myLogger
from osa.job import run_sacct, get_sacct_output
from osa.utils.utils import date_to_dir, get_calib_filters, get_lstchain_version, container_prefix
from lstchain.onsite import find_systematics_correction_file

from osa.paths import (
    catB_closed_file_exists,
    catB_calibration_file_exists,
    analysis_path,
    get_major_version,
    search_calibration_files,
)

log = myLogger(logging.getLogger())

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",                                                                                            
    "--config",                                                                                      
    action="store",
    type=Path,
    help="Configuration file",
)
parser.add_argument(                                                                                     
    "-d",                                                                                            
    "--date",
    type=valid_date,
    default=None,
)
parser.add_argument(
    "-v",                                                                                      
    "--verbose",
    action="store_true",
    default=False,
    help="Activate debugging mode.",
)
parser.add_argument(
    "-s",
    "--simulate",
    action="store_true",
    default=False,
    help="Simulate launching of the sequencer_catB_tailcuts script.",
)
parser.add_argument(
    "--overwrite-tailcuts",
    action="store_true",
    default=False,
    help="Overwrite the tailcuts config file if it already exists.",
)
parser.add_argument(
    "--overwrite-catB",
    action="store_true",
    default=False,
    help="Overwrite the Cat-B calibration files if they already exist.",
)
parser.add_argument(
    "--run-ids",
    nargs="+",
    type=int,
    default=None,
    metavar="RUN_ID",
    help="Only process these DATA run IDs (space-separated integers).",
)
parser.add_argument(
    "tel_id",
    choices=["ST", "LST1", "LST2", "all"],
    help="telescope identifier LST1, LST2, ST or all.",
)

def are_all_history_files_created(run_id: int) -> bool:
    """Check if all the history files (one per subrun) were created for a given run."""
    run_summary_dir = Path(cfg.get(options.tel_id, "RUN_SUMMARY_DIR"))
    run_summary_file = run_summary_dir / f"RunSummary_{date_to_dir(options.date)}.ecsv"
    run_summary = Table.read(run_summary_file)
    n_subruns = run_summary[run_summary["run_id"] == run_id]["n_subruns"]
    analysis_dir = Path(options.directory)
    history_files = glob.glob(f"{str(analysis_dir)}/sequence_LST1_{run_id:05d}.????.history")
    if len(history_files) == n_subruns:
        return True
    else:
        return False


def r0_to_dl1_step_finished_for_run(run_id: int) -> bool:
    """
    Check if the step r0_to_dl1 finished successfully 
    for a given run by looking the history files.
    """
    if not are_all_history_files_created(run_id):
        log.debug(f"All history files for run {run_id:05d} were not created yet.")
        return False
    analysis_dir = Path(options.directory)
    history_files = glob.glob(f"{str(analysis_dir)}/sequence_LST1_{run_id:05d}.????.history")
    for file in history_files:
        rc = Path(file).read_text().splitlines()[-1][-1]
        if rc != "0":
            print(f"r0_to_dl1 step did not finish successfully (check file {file})")
            return False
    return True


def get_catB_last_job_id(run_id: int) -> int:
    """Get job id of the last Cat-B calibration job that was launched for a given run."""
    log_dir = Path(options.directory) / "log"
    filenames = glob.glob(f"{log_dir}/catB_calibration_{run_id:05d}_*.err")
    if filenames:
        match = re.search(f"catB_calibration_{run_id:05d}_(\d+).err", sorted(filenames)[-1])
        job_id = match.group(1)
        return job_id


def launch_catB_calibration(run_id: int):
    """
    Launch the Cat-B calibration script for a given run if the Cat-B calibration 
    file has not been created yet. If the Cat-B calibration script was launched
    before and it finished successfully, it creates a catB_{run}.closed file.
    """
    job_id = get_catB_last_job_id(run_id)
    if job_id and not options.overwrite_catB:
        job_status = get_sacct_output(run_sacct(job_id=job_id))["State"]
        if job_status.item() in ["RUNNING", "PENDING"]:
            log.debug(f"Job {job_id} (corresponding to run {run_id:05d}) is still running.")

        elif job_status.item() == "COMPLETED":
            catB_closed_file = Path(options.directory) / f"catB_{run_id:05d}.closed"
            catB_closed_file.touch()
            log.debug(
                f"Cat-B job {job_id} (corresponding to run {run_id:05d}) finished "
                f"successfully. Creating file {catB_closed_file}"
            )

        else: 
            log.warning(f"Cat-B job {job_id} (corresponding to run {run_id:05d}) failed.")

    else:
        if catB_calibration_file_exists(run_id):
            if not options.overwrite_catB:
                log.info(f"Cat-B calibration file already produced for run {run_id:05d}.")
                return 
            else:
                log.info(f"Cat-B calibration file already produced for run {run_id:05d}. Overwriting it.")

        command = cfg.get("lstchain", "catB_calibration")
        if cfg.getboolean("lstchain", "use_lstcam_env_for_CatB_calib"):
            env_command = f"conda run -n lstcam-env {command}"
        else:
            env_command = command
        options.filters = get_calib_filters(run_id)
        catB_base = cfg.get(options.tel_id, "CAT_B_BASE_DIR", fallback=None)
        base_dir = Path(catB_base).resolve() if catB_base else Path(cfg.get(options.tel_id, "BASE")).resolve()
        original_base_dir = Path(cfg.get(options.tel_id, "BASE")).resolve()
        r0_dir = Path(cfg.get(options.tel_id, "R0_DIR")).resolve()
        log_dir = Path(options.directory) / "log"
        catA_calib_run = get_last_pedcalib(options.date)
        slurm_account = cfg.get("SLURM", "ACCOUNT")
        lstchain_version = get_major_version(get_lstchain_version())
        analysis_dir = cfg.get("LST1", "ANALYSIS_DIR")
        flat_date = date_to_dir(options.date)

        # Resolve Cat-A calibration and systematics files using the original base
        # dir so they are found even when CAT_B_BASE_DIR differs from BASE.
        catA_files = search_calibration_files(catA_calib_run, options.prod_id)
        if not catA_files:
            log.error(f"No Cat-A calibration file found for run {catA_calib_run}. Cannot launch Cat-B.")
            return
        catA_calib_file = catA_files[-1]
        systematics_file = find_systematics_correction_file("pro", flat_date, base_dir=original_base_dir)

        sbatch_opts = ["sbatch", f"--account={slurm_account}", "--parsable",
            "-o", f"{log_dir}/catB_calibration_{run_id:05d}_%j.out",
            "-e", f"{log_dir}/catB_calibration_{run_id:05d}_%j.err",
        ]
        inner = [env_command,
            "-r", str(run_id),
            "-b", str(base_dir),
            f"--cat_A_calibration_file={catA_calib_file}",
            f"--systematics_file={systematics_file}",
            f"--r0-dir={r0_dir}",
            f"--filters={options.filters}",
        ]

        if command=="onsite_create_cat_B_calibration_file":
            inner.append(f"--interleaved-dir={analysis_dir}")
        elif command=="lstcam_calib_onsite_create_cat_B_calibration_file":
            inner.append(f"--dl1-dir={analysis_dir}")
            inner.append(f"--lstchain-version={lstchain_version[1:]}")

        if options.overwrite_catB:
            inner.append("--yes")

        # Optionally run the command inside an apptainer image (see [lstchain]
        # apptainer_image). sbatch needs a script, so the container call is passed
        # via --wrap. Skipped when the Cat-B lstcam-env is used (not in the image).
        prefix = container_prefix()
        if prefix and not cfg.getboolean("lstchain", "use_lstcam_env_for_CatB_calib"):
            cmd = sbatch_opts + ["--wrap", shlex.join(prefix + inner)]
        else:
            cmd = sbatch_opts + inner

        if not options.simulate:
            job = sp.run(cmd, encoding="utf-8", capture_output=True, text=True, check=True)
            job_id = job.stdout.strip()
            log.debug(f"Launched Cat-B calibration job {job_id} for run {run_id}!")

        else: 
            log.info(f"Simulate launching of the {command} script.")
            

def launch_tailcuts_finder(run_id: int):
    """
    Launch the lstchain script to calculate the correct
    tailcuts to use for a given run. 
    """
    command = cfg.get("lstchain", "tailcuts_finder")
    slurm_account = cfg.get("SLURM", "ACCOUNT")
    input_dir = Path(options.directory)
    output_dir = Path(cfg.get(options.tel_id, "TAILCUTS_FINDER_DIR"))
    log_dir = Path(options.directory) / "log"
    log_file = log_dir / f"tailcuts_finder_{run_id:05d}_%j.log"
    sbatch_opts = ["sbatch", "--parsable", f"--account={slurm_account}", "-o", log_file]
    inner = [
        command,
        f"--input-dir={input_dir}",
        f"--run={run_id}",
        f"--output-dir={output_dir}",
    ]
    # Optionally run lstchain_find_tailcuts inside an apptainer image via --wrap
    # (see [lstchain] apptainer_image); unchanged host submission when unset.
    prefix = container_prefix()
    if prefix:
        cmd = sbatch_opts + ["--wrap", shlex.join(prefix + inner)]
    else:
        cmd = sbatch_opts + inner
    if not options.simulate:
        job = sp.run(cmd, encoding="utf-8", capture_output=True, text=True, check=True)
        job_id = job.stdout.strip()
        log.debug(f"Launched lstchain_find_tailcuts job {job_id} for run {run_id}!")

    else: 
        log.info(f"Simulate launching of the {command} script.")



def tailcuts_config_file_exists(run_id: int) -> bool:
    """Check if the config file created by the tailcuts finder script already exists."""
    tailcuts_config_file = Path(cfg.get(options.tel_id, "TAILCUTS_FINDER_DIR")) / f"dl1ab_Run{run_id:05d}.json"
    return tailcuts_config_file.exists()
    
        
def main():
    """
    Main script to be called as cron job. It launches the Cat-B calibration script 
    and the tailcuts finder script for each run of the corresponding date, and creates
    the catB_{run}.closed files if Cat-B calibration has finished successfully.
    """ 
    opts = parser.parse_args()
    options.tel_id = opts.tel_id
    options.simulate = opts.simulate
    options.overwrite_tailcuts = opts.overwrite_tailcuts
    options.overwrite_catB = opts.overwrite_catB
    options.date = opts.date
    options.date = set_default_date_if_needed()
    options.configfile = opts.config.resolve()
    options.directory = analysis_path(options.tel_id)
    run_ids_filter = set(opts.run_ids) if opts.run_ids is not None else None

    if opts.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    run_summary_dir = Path(cfg.get(options.tel_id, "RUN_SUMMARY_DIR"))
    run_summary = Table.read(run_summary_dir / f"RunSummary_{date_to_dir(options.date)}.ecsv")
    data_runs = run_summary[run_summary["run_type"]=="DATA"]
    for run_id in data_runs["run_id"]:
        if run_ids_filter is not None and run_id not in run_ids_filter:
            log.debug(f"Skipping run {run_id:05d} (not in --run-ids list)")
            continue
        # first check if the dl1a files are produced
        if not r0_to_dl1_step_finished_for_run(run_id):
            log.info(f"The r0_to_dl1 step did not finish yet for run {run_id:05d}. Please try again later.")
        else:
            # launch catB calibration and tailcut finder in parallel
            if cfg.getboolean("lstchain", "apply_catB_calibration") and not catB_closed_file_exists(run_id):
                if catB_calibration_file_exists(run_id):
                    # Calibration file already produced (e.g. by a previous run) but
                    # the .closed marker is missing — just create it.
                    catB_closed_file = Path(options.directory) / f"catB_{run_id:05d}.closed"
                    catB_closed_file.touch()
                    log.info(
                        f"Cat-B calibration file already exists for run {run_id:05d}. "
                        f"Created missing .closed file: {catB_closed_file}"
                    )
                else:
                    launch_catB_calibration(run_id)
            if not cfg.getboolean("lstchain", "apply_standard_dl1b_config"):
                if tailcuts_config_file_exists(run_id) and not options.overwrite_tailcuts:
                    log.debug(
                        f"Tailcuts config file already exists for run {run_id:05d}. Use --overwrite-tailcuts to overwrite it."
                    )
                else:
                    launch_tailcuts_finder(run_id)


if __name__ == "__main__":
    main()
