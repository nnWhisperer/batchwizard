# cli.py
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from openai import (
    APIError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from rich.console import Console
from rich.table import Table

from .config import BatchWizardSettings, config
from .processor import BatchProcessor
from .ui import BatchWizardUI
from .utils import get_api_key, set_api_key, setup_logger

app = typer.Typer(help="BatchWizard: Manage OpenAI batch processing jobs with ease")
console = Console()
logger = setup_logger(console)


@app.command()
def process(
    input_paths: list[Path] = typer.Argument(
        ..., help="Paths to input files or directories for processing"
    ),
    output_directory: Optional[Path] = typer.Option(
        None, help="Directory to store output files"
    ),
    max_concurrent_jobs: int = typer.Option(
        5, help="Maximum number of concurrent jobs"
    ),
    check_interval: int = typer.Option(
        5, help="Initial interval (in seconds) between job status checks"
    ),
):
    """Process batch jobs from input files or directories."""
    if not output_directory:
        output_directory = Path.cwd() / "results"
    output_directory.mkdir(parents=True, exist_ok=True)

    api_key = get_api_key()
    if not api_key:
        logger.error(
            "API key not set. Please set it using 'openaibatch configure --set-key YOUR_API_KEY'"
        )
        raise typer.Exit(code=1)

    config.settings.max_concurrent_jobs = max_concurrent_jobs
    config.settings.check_interval = check_interval
    config.save()

    processor = BatchProcessor()
    ui = BatchWizardUI(Console())

    async def run_and_close():
        try:
            await ui.run_processing(processor, input_paths, output_directory)
        finally:
            await processor.close()

    asyncio.run(run_and_close())



@app.command()
def configure(
    set_key: Optional[str] = typer.Option(
        None, "--set-key", help="Set the OpenAI API key"
    ),
    show: bool = typer.Option(False, "--show", help="Show the current configuration"),
    reset: bool = typer.Option(
        False, "--reset", help="Reset the configuration to default values"
    ),
):
    """Manage BatchWizard configuration."""
    if set_key:
        set_api_key(set_key)
        console.print("[green]API key set successfully.[/green]")
    elif show:
        api_key = get_api_key()
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if api_key else "Not set"
        console.print(f"API Key: {masked_key}")
        console.print(f"Max Concurrent Jobs: {config.settings.max_concurrent_jobs}")
        console.print(f"Check Interval: {config.settings.check_interval} seconds")
    elif reset:
        config.settings = BatchWizardSettings()
        config.save()
        console.print("[yellow]Configuration reset to default values.[/yellow]")
    else:
        console.print(
            "Use --set-key, --show, or --reset options to manage configuration."
        )


@app.command()
def list_jobs(
    limit: int = typer.Option(100, help="Number of jobs to display"),
    all: bool = typer.Option(False, "--all", help="Display all jobs"),
):
    """List recent batch jobs."""

    async def fetch_jobs():
        processor = BatchProcessor()
        console = Console()  # Create a Console object directly
        try:
            jobs = await processor.client.batches.list(limit=None if all else limit)
            table = Table(title="Batch Jobs")
            table.add_column("Job ID", style="cyan")
            table.add_column("Status", style="magenta")
            table.add_column("Created At", style="green")
            table.add_column("Completed", style="blue")
            table.add_column("Failed", style="red")

            for job in jobs.data:
                created_at = datetime.fromtimestamp(job.created_at).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                table.add_row(
                    job.id,
                    job.status,
                    created_at,
                    str(job.request_counts.completed),
                    str(job.request_counts.failed),
                )
            console.print(table)  # Use the console object to print the table
        finally:
            await processor.close()

    asyncio.run(fetch_jobs())


@app.command()
def cancel(
    job_id: str = typer.Argument(..., help="ID of the batch job to cancel"),
):
    """Cancel a specific batch job."""

    async def cancel_job():
        processor = BatchProcessor()
        try:
            await processor.client.batches.cancel(job_id)
            console.print(f"[green]Job {job_id} cancelled successfully.[/green]")
        except AuthenticationError as e:
            console.print(f"[red]Authentication Error: Invalid or expired API key.[/red]")
            console.print("[yellow]Suggestion: Please check and update your OpenAI API key using 'batchwizard configure --set-key YOUR_API_KEY'[/yellow]")
        except PermissionDeniedError as e:
            console.print(f"[red]Permission Denied: Insufficient permissions to cancel job {job_id}.[/red]")
            console.print("[yellow]Suggestion: Check your account permissions and ensure you have access to batch operations.[/yellow]")
        except RateLimitError as e:
            console.print(f"[red]Rate Limit Exceeded: Too many requests to the OpenAI API.[/red]")
            console.print("[yellow]Suggestion: Wait a moment before retrying or reduce request frequency.[/yellow]")
        except NotFoundError as e:
            console.print(f"[red]Job Not Found: Batch job {job_id} does not exist.[/red]")
            console.print("[yellow]Suggestion: Verify the job ID is correct using 'batchwizard list-jobs'.[/yellow]")
        except ConflictError as e:
            console.print(f"[red]Conflict Error: Job {job_id} cannot be cancelled in its current state.[/red]")
            console.print("[yellow]Suggestion: Check the job status - only jobs that are 'validating', 'in_progress', or 'finalizing' can be cancelled.[/yellow]")
        except InternalServerError as e:
            console.print(f"[red]OpenAI Server Error: Internal server error occurred.[/red]")
            console.print("[yellow]Suggestion: Please retry the operation later.[/yellow]")
        except APIError as e:
            console.print(f"[red]OpenAI API Error: {str(e)}[/red]")
            console.print("[yellow]Suggestion: Check the error message for specific details.[/yellow]")
        except Exception as e:
            console.print(f"[red]Unexpected Error cancelling job {job_id}: {str(e)}[/red]")
            console.print("[yellow]Suggestion: Contact support if this error persists.[/yellow]")
        finally:
            await processor.close()

    asyncio.run(cancel_job())


@app.command()
def download(
    job_id: str = typer.Argument(
        ..., help="ID of the batch job to download results for"
    ),
    output_file: Path = typer.Option(
        None, help="Path to save the output file (default: <job_id>_results.jsonl)"
    ),
):
    """Download results for a completed batch job."""
    if not output_file:
        output_file = Path(f"{job_id}_results.jsonl")

    async def download_results():
        processor = BatchProcessor()
        try:
            batch_job = await processor.client.batches.retrieve(job_id)
            if batch_job.status != "completed":
                status_messages = {
                    "validating": "Job is still being validated",
                    "failed": "Job failed during processing",
                    "in_progress": "Job is currently in progress",
                    "finalizing": "Job is being finalized",
                    "cancelled": "Job was cancelled",
                    "expired": "Job expired before completion"
                }
                status_msg = status_messages.get(batch_job.status, f"Job status: {batch_job.status}")
                console.print(f"[yellow]Job {job_id} is not completed ({status_msg}). Cannot download results.[/yellow]")
                
                if batch_job.status == "failed":
                    console.print("[yellow]Suggestion: Check the job details for failure reasons and resubmit with corrected input.[/yellow]")
                elif batch_job.status in ["validating", "in_progress", "finalizing"]:
                    console.print("[yellow]Suggestion: Wait for the job to complete before downloading results.[/yellow]")
                elif batch_job.status == "expired":
                    console.print("[yellow]Suggestion: Resubmit the job as expired jobs cannot be recovered.[/yellow]")
                return

            success = await processor.download_batch_results(batch_job, output_file)
            if success:
                console.print(
                    f"[green]Results for job {job_id} downloaded successfully to {output_file}[/green]"
                )
            else:
                console.print(f"[red]Failed to download results for job {job_id}[/red]")
                console.print("[yellow]Suggestion: Check if the output file location is writable and try again.[/yellow]")
        except AuthenticationError as e:
            console.print(f"[red]Authentication Error: Invalid or expired API key.[/red]")
            console.print("[yellow]Suggestion: Please check and update your OpenAI API key using 'batchwizard configure --set-key YOUR_API_KEY'[/yellow]")
        except PermissionDeniedError as e:
            console.print(f"[red]Permission Denied: Insufficient permissions to access job {job_id}.[/red]")
            console.print("[yellow]Suggestion: Check your account permissions and ensure you have access to this batch job.[/yellow]")
        except RateLimitError as e:
            console.print(f"[red]Rate Limit Exceeded: Too many requests to the OpenAI API.[/red]")
            console.print("[yellow]Suggestion: Wait a moment before retrying or reduce request frequency.[/yellow]")
        except NotFoundError as e:
            console.print(f"[red]Job Not Found: Batch job {job_id} does not exist.[/red]")
            console.print("[yellow]Suggestion: Verify the job ID is correct using 'batchwizard list-jobs'.[/yellow]")
        except BadRequestError as e:
            console.print(f"[red]Bad Request: Invalid request parameters for job {job_id}.[/red]")
            console.print(f"[yellow]Details: {str(e)}[/yellow]")
            console.print("[yellow]Suggestion: Check the job ID format and ensure it's valid.[/yellow]")
        except InternalServerError as e:
            console.print(f"[red]OpenAI Server Error: Internal server error occurred.[/red]")
            console.print("[yellow]Suggestion: Please retry the operation later.[/yellow]")
        except APIError as e:
            console.print(f"[red]OpenAI API Error: {str(e)}[/red]")
            console.print("[yellow]Suggestion: Check the error message for specific details.[/yellow]")
        except Exception as e:
            console.print(f"[red]Unexpected Error downloading results for job {job_id}: {str(e)}[/red]")
            console.print("[yellow]Suggestion: Check file permissions and disk space, or contact support if this error persists.[/yellow]")
        finally:
            await processor.close()

    asyncio.run(download_results())


if __name__ == "__main__":
    app()



