"""Own the lifetime of each checkpoint's Docker environment."""
from contextlib import contextmanager

from . import checkpoints


class CheckpointEnvironments:
    def __init__(self, state, settings):
        self.state = state
        self.settings = settings

    @contextmanager
    def open(self, identity):
        try:
            metadata = checkpoints.start(self.state, identity, self.settings)
            yield f"http://127.0.0.1:{metadata['backend_port']}/artifactnet"
        except BaseException as error:
            # Preserve the original failure if Docker also fails during cleanup.
            try:
                checkpoints.stop(self.state, identity, remove_data=True)
            except Exception as cleanup_error:
                error.add_note(f"Environment cleanup also failed: {cleanup_error}")
            raise
        else:
            checkpoints.stop(self.state, identity, remove_data=True)
