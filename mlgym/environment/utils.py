"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Utility functions for MLGym environment management.

This module provides utility functions for container management, file operations,
and process control. It handles Docker/Podman container interactions, file copying,
and process monitoring with timeout capabilities.

Adapted from SWE-agent/sweagent/environment/utils.py
"""

from __future__ import annotations

import os
import platform
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from subprocess import PIPE, STDOUT
from typing import TYPE_CHECKING

import docker
import docker.errors
import docker.types
from docker.models.containers import Container

from mlgym.exceptions import NoOutputTimeoutError
from mlgym.utils.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger("env_utils")


DOCKER_START_UP_DELAY = float(os.getenv("MLGYM_DOCKER_START_UP_DELAY", "1"))
PROCESS_DONE_MARKER_START = "///PROCESS-DONE:"
PROCESS_DONE_MARKER_END = ":PROCESS-DONE///"
PROCESS_DONE_REGEX = re.compile(rf"{PROCESS_DONE_MARKER_START}(.+?){PROCESS_DONE_MARKER_END}")
TQDM_PATTERN = re.compile(r"(?:.*?:\s*)?(\d+%\|.*?\|.*\[.*?\])")

# ! NOTE: Most of this file can be moved to containerization module. Issue #20


# FIXME: Move to containerization module and fix complexity. Issue #20
def read_with_timeout(  # noqa: C901
    container: subprocess.Popen, timeout_duration: float, no_output_timeout_duration: float
) -> tuple[str, str]:
    """
    Read data from a subprocess with timeout constraints.

    Uses file descriptors to read data from subprocess in a non-blocking way.
    Filters out tqdm progress bars and handles process completion markers.

    Args:
        container: Subprocess container to read from
        timeout_duration: Maximum total execution time in seconds
        no_output_timeout_duration: Maximum time without output in seconds

    Returns:
        tuple[str, str]: Tuple containing:
            - output: Filtered output from subprocess
            - exit_code: Process exit code as string

    Raises:
        TimeoutError: If timeout_duration is reached
        NoOutputTimeoutError: If no_output_timeout_duration is reached
        RuntimeError: If subprocess exits unexpectedly
        ValueError: If process completion marker not found
    """
    buffer = b""
    fd = container.stdout.fileno()  # type: ignore
    start_time = time.time()
    end_time = start_time + timeout_duration
    end_time_no_output = start_time + no_output_timeout_duration

    # Select is not available on windows
    is_windows = platform.system() == "Windows"
    if not is_windows:
        import select
    else:
        os.set_blocking(fd, False)

    def ready_to_read(fd: int) -> bool:
        if is_windows:
            # We can't do the extra check
            return True
        return bool(select.select([fd], [], [], 0.01)[0])

    process_done = False

    while time.time() < min(end_time, end_time_no_output):
        if ready_to_read(fd):
            try:
                data = os.read(fd, 4096)
            except BlockingIOError:
                # logger.error("BlockingIOError while reading from subprocess.", exc_info=True)
                break
            if data:
                end_time_no_output = time.time() + no_output_timeout_duration
                buffer += data
                decoded = buffer.decode("utf-8", errors="backslashreplace").replace("\r\n", "\n")

                if PROCESS_DONE_MARKER_START in decoded:
                    process_done = True
                    break
        time.sleep(0.01)  # Prevents CPU hogging

    decoded = buffer.decode("utf-8", errors="backslashreplace").replace("\r\n", "\n")
    filtered_lines = []

    # Filter out tqdm progress bars
    for line in decoded.splitlines():
        if not TQDM_PATTERN.search(line) and not line.startswith(PROCESS_DONE_MARKER_START) and line != "":
            filtered_lines.append(line)
    # filtered_lines = decoded.splitlines()

    body = "\n".join(line for line in filtered_lines)

    if container.poll() is not None:
        msg = f"Subprocess exited unexpectedly.\nCurrent buffer: {buffer.decode()}"
        raise RuntimeError(msg)

    current_time = time.time()
    if not process_done and current_time >= min(end_time, end_time_no_output):
        if current_time >= end_time:
            msg = f"Timeout reached while reading from subprocess.\nCurrent buffer: {buffer.decode()}"
            raise TimeoutError(msg)
        else:
            msg = f"No output timeout reached while reading from subprocess.\nCurrent buffer: {buffer.decode()}"
            raise NoOutputTimeoutError(msg, body)

    _results = PROCESS_DONE_REGEX.search(decoded)
    if _results is None:
        msg = f"Could not find process done marker in last line: {decoded=}, {body=}"
        raise ValueError(msg)
    exit_code = _results.group(1)
    return body.replace(f"{PROCESS_DONE_MARKER_START}{exit_code}{PROCESS_DONE_MARKER_END}", ""), exit_code


# FIXME: Move to containerization module. Issue #20
def read_with_timeout_pid(container: subprocess.Popen, pid_func: Callable, timeout_duration: float) -> str:
    """
    Read data from subprocess while monitoring process IDs.

    Reads output while checking for specific process IDs to be completed.
    Handles platform-specific blocking behavior.

    Args:
        container: Subprocess container to read from
        pid_func: Function that returns list of process IDs to monitor
        timeout_duration: Maximum execution time in seconds

    Returns:
        str: Data read from subprocess, stripped of trailing newlines

    Raises:
        TimeoutError: If timeout_duration is reached
        RuntimeError: If subprocess exits unexpectedly
    """
    buffer = b""
    fd = container.stdout.fileno()  # type: ignore
    end_time = time.time() + timeout_duration

    # Select is not available on windows
    is_windows = platform.system() == "Windows"
    if not is_windows:
        import select
    else:
        os.set_blocking(fd, False)

    def ready_to_read(fd: int) -> bool:
        if is_windows:
            # We can't do the extra check
            return True
        return bool(select.select([fd], [], [], 0.01)[0])

    while time.time() < end_time:
        pids = pid_func()
        if len(pids) > 0:
            # There are still PIDs running
            time.sleep(0.05)
            continue
        if ready_to_read(fd):
            data = os.read(fd, 4096)
            if data:
                buffer += data
        else:
            # No more data to read
            break
        time.sleep(0.05)  # Prevents CPU hogging

    if container.poll() is not None:
        msg = f"Subprocess exited unexpectedly.\nCurrent buffer: {buffer.decode()}"
        raise RuntimeError(msg)
    if time.time() >= end_time:
        msg = f"Timeout reached while reading from subprocess.\nCurrent buffer: {buffer.decode()}\nRunning PIDs: {pids}"
        raise TimeoutError(msg)
    return buffer.decode()


def copy_file_to_container(container: Container, container_type: str, contents: str, container_path: str) -> None:
    """
    Copy file contents to container.

    Creates a temporary file with given contents and copies it to specified
    path in container. Handles cleanup of temporary files.

    Args:
        container: Docker/Podman container object
        container_type: Type of container ("docker" or "podman")
        contents: String contents to write to file
        container_path: Destination path in container

    Raises:
        RuntimeError: If copy operation fails
    """
    temp_file_name = None

    try:
        assert isinstance(container, Container)
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_name = temp_file.name
            # Write the string to the temporary file and ensure it's written to disk
            temp_file.write(contents.encode("utf-8"))
            temp_file.flush()
            os.fsync(temp_file.fileno())

        copy_anything_to_container(container, container_type, temp_file.name, container_path)
    except Exception:
        logger.exception("An error occurred while copying file to container.")
    finally:
        if temp_file_name and Path(temp_file_name).exists():
            os.remove(temp_file_name)


def copy_anything_to_container(container: Container, container_type: str, host_path: str, container_path: str) -> None:
    """
    Copy files or directories from host to container.

    Handles file ownership and permissions in container after copy.
    Supports both files and directories.

    Args:
        container: Docker/Podman container object
        container_type: Type of container ("docker" or "podman")
        host_path: Source path on host
        container_path: Destination path in container

    Raises:
        FileNotFoundError: If host_path doesn't exist
        RuntimeError: If copy operation fails
    """
    if not Path(host_path).exists():
        msg = f"Path {host_path} does not exist, cannot copy it to container."
        raise FileNotFoundError(msg)

    # cannot directly use -a flag because some machines require the user in docker container to be present on host machine. We will handle the ownership in _setup_workspace file.
    cmd = [container_type, "cp", host_path, f"{container.id}:{container_path}"]
    logger.debug(f"Copying {host_path} to container at {container_path} with command: {shlex.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        msg = f"Error copying {host_path} to container at {container_path}: {e}"
        raise RuntimeError(msg) from e
    # set agent as the owner of the copied files
    assert container.id is not None
    cmd = [container_type, "exec", "-u", "root", container.id, "chown", "-R", "agent:mlgym", container_path]
    logger.debug(f"Setting agent as owner of copied files at {container_path} with command: {shlex.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        msg = f"Error setting agent as owner of copied files at {container_path}: {e}"
        raise RuntimeError(msg) from e


def copy_anything_from_container(
    container: Container, container_type: str, host_path: str, container_path: str
) -> None:
    """
    Copy files or directories from container to host.

    Args:
        container: Docker/Podman container object
        host_path: Destination path on host
        container_path: Source path in container

    Raises:
        RuntimeError: If copy operation fails
    """
    assert isinstance(container, Container)
    # use -a to set ownership of the copied files to the user in the container
    cmd = [container_type, "cp", f"{container.id}:{container_path}", host_path]
    logger.debug(f"Copying {container_path} to host at {host_path} with command: {shlex.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        msg = f"Error copying {container_path} from container at {host_path}: {e}"
        raise RuntimeError(msg) from e


### Docker Container Utilities ###


def get_container(
    container_name: str,
    image_name: str,
    container_type: str,
    devices: list[str],
    persistent: bool = False,
    container_mounts: list[str] | None = None,
) -> tuple[subprocess.Popen, set]:
    """
    Get a container instance with specified configuration.

    Creates or retrieves a container with given parameters. Handles both
    persistent and non-persistent containers.

    Args:
        container_name: Name for the container
        image_name: Docker/Podman image to use
        container_type: Type of container ("docker" or "podman")
        devices: List of devices (e.g., GPUs) to make available
        persistent: Whether container should persist between runs
        container_mounts: List of host paths to mount in container

    Returns:
        tuple[subprocess.Popen, set]: Tuple containing:
            - Container process object
            - Set of process IDs to monitor

    Raises:
        RuntimeError: If image not found or container creation fails
    """
    if container_mounts is None:
        container_mounts = []
    if not image_exists(image_name):
        msg = (
            f"Image {image_name} not found. Please ensure it is built and available. "
            "Please double-check that you followed all installation/setup instructions from the "
            "readme."
        )
        raise RuntimeError(msg)
    if persistent:
        return _get_persistent_container(
            container_name, image_name, container_type, container_mounts=container_mounts, devices=devices
        )
    else:
        return _get_non_persistent_container(
            container_name, image_name, container_type, container_mounts=container_mounts, devices=devices
        )


def image_exists(image_name: str) -> bool:
    """
    Check if Docker image exists locally.

    Args:
        image_name: Name of image to check

    Returns:
        bool: True if image exists, False otherwise

    Raises:
        RuntimeError: If Docker daemon not running
    """
    try:
        client = docker.from_env()
    except docker.errors.DockerException as e:
        docker_not_running = any(
            (
                "connection aborted" in str(e).lower(),
                "connection refused" in str(e).lower(),
                "error while fetching server api version" in str(e).lower(),
            ),
        )
        if docker_not_running:
            msg = (
                "Probably the Docker daemon is not running. Please start the Docker daemon and try again. "
                "If Docker issues persist, good luck! we don't have documentation right now 😭"
            )
            raise RuntimeError(msg) from e
        raise
    filterred_images = client.images.list(filters={"reference": image_name})
    if len(filterred_images) == 0:
        return False
    elif len(filterred_images) > 1:
        RuntimeError(f"Multiple images found for {image_name}, that's weird.")
    attrs = filterred_images[0].attrs
    if attrs is not None:
        logger.info(
            f"Found image {image_name} with tags: {attrs['RepoTags']}, created: {attrs['Created']} "
            f"for {attrs['Os']} {attrs['Architecture']}.",
        )
    return True


def get_background_pids(container_obj: Container) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """
    Get background process IDs in container.

    Args:
        container_obj: Docker/Podman container object

    Returns:
        tuple[list[tuple[int, str]], list[tuple[int, str]]]: Tuple containing:
            - List of bash process (pid, name) tuples
            - List of other process (pid, name) tuples
    """
    pids = container_obj.exec_run("ps -eo pid,comm --no-headers").output.decode().split("\n")
    pids = [x.split() for x in pids if x]
    pids = [x for x in pids if x[1] not in {"ps"} and x[0] != "1"]
    bash_pids = [x for x in pids if x[1] == "bash"]
    other_pids = [x for x in pids if x[1] not in {"bash"}]
    return bash_pids, other_pids


def _get_container_mounts_list(container_mounts: list[str]) -> list[docker.types.Mount]:
    """
    Convert container mount paths to Docker mount objects.

    Args:
        container_mounts: List of host paths to mount

    Returns:
        list[docker.types.Mount]: List of Docker mount objects

    Note:
        Skips invalid mount paths with error logging
    """
    initialized_mounts = []
    try:
        for container_path in container_mounts:
            path = Path(container_path).absolute()
            if path.is_dir():
                initialized_mounts.append(docker.types.Mount(source=str(path), target=f"/{path.name}"))
    except Exception:
        logger.exception("Failed to process container mounts, skipping mount.")
        return []
    else:
        return initialized_mounts


def _get_non_persistent_container(
    container_name: str, image_name: str, container_type: str, container_mounts: list[str], devices: list[str]
) -> tuple[subprocess.Popen, set]:
    """
    Create non-persistent container.

    Args:
        container_name: Name for container
        image_name: Docker/Podman image to use
        container_type: Type of container ("docker" or "podman")
        container_mounts: List of host paths to mount
        devices: List of devices to make available

    Returns:
        tuple[subprocess.Popen, set]: Tuple containing:
            - Container process object
            - Set of process IDs to monitor

    Raises:
        RuntimeError: If container creation fails
    """
    if len(devices) == 1 and "cpu" in devices[0]:
        startup_cmd = [
            container_type,
            "run",
            "-i",
            "--rm",
            "--network",
            "host",
            *[item for mount in container_mounts for item in ("-v", f"{Path(mount).absolute()}:/{Path(mount).name}")],
            "--name",
            container_name,
            image_name,
            "/bin/bash",
            "-l",
        ]
    else:
        device_str = f'"device={",".join(devices)}"' if container_type == "docker" else f"{','.join(devices)}"
        startup_cmd = [
            container_type,
            "run",
            "-i",
            "--rm",
            "--network",
            "host",
            "--gpus",
            device_str,
            *[item for mount in container_mounts for item in ("-v", f"{Path(mount).absolute()}:/{Path(mount).name}")],
            "--name",
            container_name,
            image_name,
            "/bin/bash",
            "-l",
        ]
    logger.debug("Starting container with command: %s", shlex.join(startup_cmd))
    container = subprocess.Popen(
        startup_cmd,
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
        bufsize=1,  # line buffered
    )
    time.sleep(DOCKER_START_UP_DELAY)
    # try to read output from container setup (usually an error), timeout if no output
    output = read_with_timeout_pid(container, lambda: [], timeout_duration=2)
    if output:
        logger.error(f"Unexpected container setup output: {output}")
    # bash PID is always 1 for non-persistent containers
    return container, {
        "1",
    }


# FIXME: Move to containerization module and fix complexity. Issue #20
def _get_persistent_container(  # noqa: C901
    container_name: str,
    image_name: str,
    container_type: str,
    container_mounts: list[str],
    devices: list[str],
    persistent: bool = False,
) -> tuple[subprocess.Popen, set]:
    """
    Create or retrieve persistent container.

    Handles container lifecycle states (created, running, exited, paused)
    and initializes new container if needed.

    Args:
        container_name: Name for container
        image_name: Docker/Podman image to use
        container_type: Type of container ("docker" or "podman")
        container_mounts: List of host paths to mount
        devices: List of devices to make available
        persistent: Whether container should persist between runs

    Returns:
        tuple[subprocess.Popen, set]: Tuple containing:
            - Container process object
            - Set of process IDs to monitor

    Raises:
        RuntimeError: If container in unexpected state or setup fails
    """
    client = docker.from_env()
    containers = client.containers.list(all=True, filters={"name": container_name})
    if container_name in [c.name for c in containers]:
        container_obj = client.containers.get(container_name)
        if container_obj.status in {"created"}:
            container_obj.start()
        elif container_obj.status in {"running"}:
            pass
        elif container_obj.status in {"exited"}:
            container_obj.restart()
        elif container_obj.status in {"paused"}:
            container_obj.unpause()
        else:
            msg = f"Unexpected container status: {container_obj.status}"
            raise RuntimeError(msg)
    else:
        initialized_mounts = _get_container_mounts_list(container_mounts)
        container_obj = client.containers.run(
            image_name,
            command="/bin/bash -l -m",
            name=container_name,
            stdin_open=True,
            tty=True,
            detach=True,
            auto_remove=not persistent,
            mounts=initialized_mounts,
        )
        container_obj.start()
    startup_cmd = [
        container_type,
        "exec",
        "-i",
        container_name,
        "/bin/bash",
        "-l",
    ]
    logger.debug("Starting container with command: %s", shlex.join(startup_cmd))
    container = subprocess.Popen(
        startup_cmd,
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
        bufsize=1,  # line buffered
    )
    time.sleep(DOCKER_START_UP_DELAY)
    # try to read output from container setup (usually an error), timeout if no output
    output = read_with_timeout_pid(container, lambda: [], timeout_duration=2)
    if output:
        logger.error(f"Unexpected container setup output: {output}")
    # Get the process IDs of the container
    # There should be at least a head process and possibly one child bash process
    bash_pids, other_pids = get_background_pids(container_obj)
    total_time_slept = DOCKER_START_UP_DELAY
    # Let's wait for a maximum of 5 x DOCKER_START_UP_DELAY seconds
    # and then check again.
    while len(bash_pids) > 1 or len(other_pids) > 0:
        time.sleep(1)
        total_time_slept += 1
        bash_pids, other_pids = get_background_pids(container_obj)
        if total_time_slept > 5 * DOCKER_START_UP_DELAY:
            break
    bash_pid = 1
    if len(bash_pids) == 1:
        bash_pid = bash_pids[0][0]
    elif len(bash_pids) > 1 or len(other_pids) > 0:
        msg = (
            "Detected alien processes attached or running. Please ensure that no other agents "
            f"are running on this container. PIDs: {bash_pids}, {other_pids}"
        )
        raise RuntimeError(msg)
    return container, {str(bash_pid), "1"}
