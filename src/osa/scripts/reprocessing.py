"""Script to reprocess a list of dates without overwhelming the job system manager."""

import logging
import subprocess as sp
import time
from collections import defaultdict
from pathlib import Path

import click

from osa.configs.config import DEFAULT_CFG
from osa.utils.logging import myLogger
from osa.utils.utils import wait_for_daytime

log = myLogger(logging.getLogger(__name__))


def number_of_pending_jobs():
    """Return the number of jobs in the slurm queue."""
    cmd = ["squeue", "-u", "lstanalyzer", "-h", "-t", "pending", "-r"]
    output = sp.check_output(cmd)
    return output.count(b"\n")


def run_script(
    script: str,
    date,
    config: Path,
    no_dl2: bool,
    no_gainsel: bool,
    no_calib: bool,
    no_dl1ab: bool,
    simulate: bool,
    force: bool,
    overwrite_tailcuts: bool,
    overwrite_catb: bool,
    run_ids=None,
):
    """Run the sequencer for a given date, optionally restricted to specific run IDs."""
    osa_config = Path(config).resolve()

    cmd = [script, "--config", str(osa_config), "--date", date]

    if no_dl2:
        cmd.append("--no-dl2")

    if no_gainsel:
        cmd.append("--no-gainsel")

    if no_calib:
        cmd.append("--no-calib")

    if no_dl1ab:
        cmd.append("--no-dl1ab")

    if simulate:
        cmd.append("--simulate")

    if force:
        cmd.append("--force")

    if overwrite_tailcuts:
        cmd.append("--overwrite-tailcuts")

    if overwrite_catb:
        cmd.append("--overwrite-catB")

    if run_ids:
        cmd.append("--run-ids")
        cmd.extend(str(r) for r in sorted(run_ids))

    # Append the telescope to the command in the last place
    cmd.append("LST1")

    log.info(f"\nRunning {' '.join(cmd)}")
    sp.run(cmd)


def check_job_status_and_wait(max_jobs=2500):
    """Check the status of the jobs in the queue and wait for them to finish."""
    while number_of_pending_jobs() > max_jobs:
        log.info("Waiting 2 hours for slurm queue to decrease...")
        time.sleep(7200)


def get_list_of_dates(dates_file):
    """Read the files with the dates to be processed and build a list of dates."""
    with open(dates_file, "r") as file:
        list_of_dates = [line.strip() for line in file]
    return list_of_dates


def get_runs_per_date(runs_file) -> dict:
    """
    Read a file mapping dates to run IDs and return a dict {date: set[int]}.

    Expected format — one entry per line, either:
      YYYY-MM-DD                         (all runs for that date)
      YYYY-MM-DD RUN1 RUN2 ...           (space-separated run IDs after the date)
      YYYY-MM-DD RUN1,RUN2,...           (comma-separated run IDs after the date)

    Lines starting with '#' are ignored.
    """
    runs_per_date: dict[str, set] = defaultdict(set)
    with open(runs_file) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            # Replace commas with spaces so both separators work uniformly
            parts = line.replace(",", " ").split()
            date = parts[0]
            if len(parts) == 1:
                # No run IDs specified → process all runs for this date (None sentinel)
                runs_per_date[date]  # creates the key with an empty set
            else:
                for token in parts[1:]:
                    runs_per_date[date].add(int(token))
    # Convert empty sets back to None so downstream code knows "all runs"
    return {date: (ids if ids else None) for date, ids in runs_per_date.items()}


@click.command()
@click.option("--no-dl2", is_flag=True, help="Do not run the DL2 step.")
@click.option("--no-gainsel", is_flag=True, help="Do not require gain selection to be finished.")
@click.option("--no-calib", is_flag=True, help="Do not run the calibration step.")
@click.option("--no-dl1ab", is_flag=True, help="Do not run the DL1AB step.")
@click.option("-s", "--simulate", is_flag=True, help="Activate simulation mode.")
@click.option("-f", "--force", is_flag=True, help="Force the autocloser to close the day.")
@click.option(
    "--overwrite-tailcuts",
    is_flag=True,
    help="Overwrite the tailcuts config file if it already exists.",
)
@click.option(
    "--overwrite-catB",
    is_flag=True,
    help="Overwrite the Cat-B calibration files if they already exist.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    default=DEFAULT_CFG,
    help="Path to the OSA config file.",
)
@click.argument(
    "script",
    type=click.Choice(
        ["sequencer", "closer", "copy_datacheck", "autocloser", "sequencer_catB_tailcuts"]
    ),
)
@click.argument("dates-file", type=click.Path(exists=True))
def main(
    script: str = None,
    dates_file: Path = None,
    config: Path = DEFAULT_CFG,
    no_dl2: bool = False,
    no_gainsel: bool = False,
    no_calib: bool = False,
    no_dl1ab: bool = False,
    simulate: bool = False,
    force: bool = False,
    overwrite_tailcuts: bool = False,
    overwrite_catb: bool = False,
):
    """
    Loop over the dates (and optionally run IDs) listed in the input file and
    launch SCRIPT for each of them.

    \b
    The input file supports two formats:

      Simple dates file (one date per line):
        2024-01-15
        2024-01-16

      Runs file (date followed by run IDs):
        2024-01-15 3456 3457 3458
        2024-01-16 3501,3502
        2024-01-17              # no run IDs = all runs for that date

    When run IDs are present the sequencer is called with --run-ids so that
    only those specific DATA runs are submitted (the calibration sequence is
    always included as a SLURM dependency).
    """
    logging.basicConfig(level=logging.INFO)

    runs_per_date = get_runs_per_date(dates_file)

    # Check slurm queue status
    check_job_status_and_wait()

    for date, run_ids in runs_per_date.items():
        # Avoid running jobs while it is still night time
        wait_for_daytime()

        if run_ids:
            log.info(f"Processing date {date} with run IDs: {sorted(run_ids)}")
        else:
            log.info(f"Processing date {date} (all runs)")

        run_script(
            script,
            date,
            config,
            no_dl2,
            no_gainsel,
            no_calib,
            no_dl1ab,
            simulate,
            force,
            overwrite_tailcuts,
            overwrite_catb,
            run_ids=run_ids,
        )
        log.info("Waiting 1 minute to launch the process for the next date...\n")
        time.sleep(60)

        # Check slurm queue status and sleep for a while to avoid overwhelming the queue
        check_job_status_and_wait()

    log.info("Done! No more dates to process.")


if __name__ == "__main__":
    main()
