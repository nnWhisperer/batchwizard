# processor.py
import asyncio
from pathlib import Path
from typing import List, Optional

import aiofiles
from loguru import logger
from openai import (
    APIError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from .config import config
from .models import BatchJob, BatchJobResult


class BatchProcessor:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.get_api_key())
        self.settings = config.settings

    async def upload_file(self, file_path: Path) -> Optional[str]:
        try:
            async with aiofiles.open(file_path, "rb") as file:
                file_content = await file.read()

            response = await self.client.files.create(
                file=(file_path.name, file_content), purpose="batch"
            )
            logger.info(
                f"File uploaded successfully: {response.id}, Filename: {file_path.name}"
            )
            return response.id
        except AuthenticationError as e:
            logger.error(f"Authentication error uploading file {file_path.name}: Invalid or expired API key")
            return None
        except PermissionDeniedError as e:
            logger.error(f"Permission denied uploading file {file_path.name}: Insufficient permissions or credits")
            return None
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded uploading file {file_path.name}: Too many requests, please wait and retry")
            return None
        except BadRequestError as e:
            logger.error(f"Bad request uploading file {file_path.name}: Invalid file format or parameters - {str(e)}")
            return None
        except InternalServerError as e:
            logger.error(f"OpenAI server error uploading file {file_path.name}: Please retry later")
            return None
        except APIError as e:
            logger.error(f"OpenAI API error uploading file {file_path.name}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading file {file_path.name}: {str(e)}")
            return None

    async def create_batch_job(self, input_file_id: str) -> Optional[BatchJob]:
        try:
            batch_job = await self.client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            logger.info(f"Created batch job with ID: {batch_job.id}")
            return BatchJob(
                id=batch_job.id,
                status=self.normalize_status(batch_job.status),
                input_file_id=input_file_id,
            )
        except AuthenticationError as e:
            logger.error(f"Authentication error creating batch job: Invalid or expired API key")
            return None
        except PermissionDeniedError as e:
            logger.error(f"Permission denied creating batch job: Insufficient permissions or credits")
            return None
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded creating batch job: Too many requests, please wait and retry")
            return None
        except BadRequestError as e:
            logger.error(f"Bad request creating batch job: Invalid parameters or file format - {str(e)}")
            return None
        except NotFoundError as e:
            logger.error(f"File not found creating batch job: Input file ID {input_file_id} not found")
            return None
        except InternalServerError as e:
            logger.error(f"OpenAI server error creating batch job: Please retry later")
            return None
        except APIError as e:
            logger.error(f"OpenAI API error creating batch job: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating batch job: {str(e)}")
            return None

    async def check_batch_status(self, batch_id: str) -> Optional[str]:
        try:
            batch_job = await self.client.batches.retrieve(batch_id)
            return self.normalize_status(batch_job.status)
        except AuthenticationError as e:
            logger.error(f"Authentication error checking batch status: Invalid or expired API key")
            return None
        except PermissionDeniedError as e:
            logger.error(f"Permission denied checking batch status: Insufficient permissions")
            return None
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded checking batch status: Too many requests, please wait and retry")
            return None
        except NotFoundError as e:
            logger.error(f"Batch job not found: Job ID {batch_id} does not exist")
            return None
        except InternalServerError as e:
            logger.error(f"OpenAI server error checking batch status: Please retry later")
            return None
        except APIError as e:
            logger.error(f"OpenAI API error checking batch status: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error checking batch status: {str(e)}")
            return None

    def normalize_status(self, status: str) -> str:
        """Normalize the status string to lowercase with underscores."""
        return status.lower().replace(" ", "_")

    async def download_batch_results(self, batch_job, output_file_path: Path) -> bool:
        try:
            if batch_job.status == "completed" and batch_job.output_file_id:
                result = await self.client.files.content(batch_job.output_file_id)
                async with aiofiles.open(output_file_path, "wb") as file:
                    await file.write(result.content)
                logger.info(f"Downloaded results to {output_file_path}")
                return True
            else:
                logger.warning(
                    f"Batch job not completed or missing output file. Status: {batch_job.status}"
                )
                return False
        except AuthenticationError as e:
            logger.error(f"Authentication error downloading batch results: Invalid or expired API key")
            return False
        except PermissionDeniedError as e:
            logger.error(f"Permission denied downloading batch results: Insufficient permissions")
            return False
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded downloading batch results: Too many requests, please wait and retry")
            return False
        except NotFoundError as e:
            logger.error(f"Output file not found: File ID {batch_job.output_file_id} does not exist")
            return False
        except InternalServerError as e:
            logger.error(f"OpenAI server error downloading batch results: Please retry later")
            return False
        except APIError as e:
            logger.error(f"OpenAI API error downloading batch results: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading batch results: {str(e)}")
            return False

    async def process_batch_job(
        self, batch_job: BatchJob, output_dir: Path
    ) -> BatchJobResult:
        check_interval = self.settings.check_interval
        while True:
            status = await self.check_batch_status(batch_job.id)
            if status == "completed":
                try:
                    batch_job = await self.client.batches.retrieve(batch_job.id)
                    if batch_job.output_file_id:
                        output_file = output_dir / f"{batch_job.id}_results.jsonl"
                        if await self.download_batch_results(batch_job, output_file):
                            logger.info(
                                f"Successfully processed batch job {batch_job.id}"
                            )
                            return BatchJobResult(
                                job_id=batch_job.id,
                                success=True,
                                output_file_path=output_file,
                            )
                    else:
                        logger.error(
                            f"No output file ID found for completed batch job {batch_job.id}"
                        )
                        return BatchJobResult(
                            job_id=batch_job.id,
                            success=False,
                            error_type="missing_output",
                            error_message="Batch job completed but no output file was generated",
                            error_details={"suggestion": "Check if the input file contained valid requests"}
                        )
                except AuthenticationError as e:
                    logger.error(f"Authentication error processing batch job {batch_job.id}: Invalid or expired API key")
                    return BatchJobResult(
                        job_id=batch_job.id,
                        success=False,
                        error_type="authentication",
                        error_message="Invalid or expired API key",
                        error_details={"suggestion": "Please check and update your OpenAI API key"}
                    )
                except PermissionDeniedError as e:
                    logger.error(f"Permission denied processing batch job {batch_job.id}: Insufficient permissions")
                    return BatchJobResult(
                        job_id=batch_job.id,
                        success=False,
                        error_type="permission_denied",
                        error_message="Insufficient permissions or credits",
                        error_details={"suggestion": "Check your account permissions and available credits"}
                    )
                except RateLimitError as e:
                    logger.error(f"Rate limit exceeded processing batch job {batch_job.id}: Too many requests")
                    return BatchJobResult(
                        job_id=batch_job.id,
                        success=False,
                        error_type="rate_limit",
                        error_message="Rate limit exceeded",
                        error_details={"suggestion": "Wait before retrying or reduce request frequency"}
                    )
                except NotFoundError as e:
                    logger.error(f"Batch job not found: {batch_job.id}")
                    return BatchJobResult(
                        job_id=batch_job.id,
                        success=False,
                        error_type="not_found",
                        error_message="Batch job not found",
                        error_details={"suggestion": "Verify the job ID is correct"}
                    )
                except InternalServerError as e:
                    logger.error(f"OpenAI server error processing batch job {batch_job.id}: Please retry later")
                    return BatchJobResult(
                        job_id=batch_job.id,
                        success=False,
                        error_type="server_error",
                        error_message="OpenAI server error",
                        error_details={"suggestion": "Please retry later"}
                    )
                except APIError as e:
                    logger.error(f"OpenAI API error processing batch job {batch_job.id}: {str(e)}")
                    return BatchJobResult(
                        job_id=batch_job.id,
                        success=False,
                        error_type="api_error",
                        error_message=str(e),
                        error_details={"suggestion": "Check the error message for specific details"}
                    )
                except Exception as e:
                    logger.error(f"Unexpected error processing batch job {batch_job.id}: {str(e)}")
                    return BatchJobResult(
                        job_id=batch_job.id,
                        success=False,
                        error_type="unexpected_error",
                        error_message=str(e),
                        error_details={"suggestion": "Contact support if this error persists"}
                    )
            elif status in ["failed", "expired", "cancelled"]:
                logger.error(f"Batch job {batch_job.id} {status}")
                error_messages = {
                    "failed": "Batch job failed during processing",
                    "expired": "Batch job expired before completion",
                    "cancelled": "Batch job was cancelled"
                }
                error_suggestions = {
                    "failed": "Check input file format and content for errors",
                    "expired": "Resubmit the job or increase completion window",
                    "cancelled": "Resubmit the job if cancellation was unintended"
                }
                return BatchJobResult(
                    job_id=batch_job.id,
                    success=False,
                    error_type="job_" + status,
                    error_message=error_messages.get(status, f"Job {status}"),
                    error_details={"suggestion": error_suggestions.get(status, "Please retry")}
                )
            elif status is None:
                logger.error(f"Failed to retrieve status for batch job {batch_job.id}")
                return BatchJobResult(
                    job_id=batch_job.id,
                    success=False,
                    error_type="status_check_failed",
                    error_message="Failed to retrieve job status",
                    error_details={"suggestion": "Check network connection and API key validity"}
                )

            await asyncio.sleep(check_interval)
            check_interval = min(
                check_interval * 1.5, 60
            )  # Implement exponential backoff

    async def process_inputs(
        self, input_paths: List[Path], output_dir: Path
    ) -> List[BatchJobResult]:
        input_files = []
        for path in input_paths:
            if path.is_dir():
                input_files.extend(path.glob("*.jsonl"))
            elif path.suffix.lower() == ".jsonl":
                input_files.append(path)
            else:
                logger.warning(f"Skipping non-JSONL file: {path}")

        if not input_files:
            logger.warning("No input files found in the provided paths")
            return []

        semaphore = asyncio.Semaphore(self.settings.max_concurrent_jobs)

        async def process_file(input_file: Path) -> Optional[BatchJobResult]:
            async with semaphore:
                file_id = await self.upload_file(input_file)
                if file_id:
                    batch_job = await self.create_batch_job(file_id)
                    if batch_job:
                        return await self.process_batch_job(batch_job, output_dir)
            return None

        tasks = [process_file(file) for file in input_files]
        results = await asyncio.gather(*tasks)
        return [result for result in results if result is not None]

    async def close(self):
        await self.client.close()






