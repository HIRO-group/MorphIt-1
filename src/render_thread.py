import threading
import queue
import time
from typing import Optional


class RenderThread:
    """
    A class to handle rendering operations in a separate thread.
    """

    def __init__(self, model, render_interval=5):
        """
        Initialize the render thread.

        Args:
            model: The SpherePacker model to render
            render_interval: How often to render frames (in iterations)
        """
        self.model = model
        self.render_interval = render_interval
        self.queue = queue.Queue()
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """Start the rendering thread."""
        if self.thread is not None and self.thread.is_alive():
            return

        self.running = True
        self.thread = threading.Thread(target=self._render_loop)
        self.thread.daemon = True  # Thread will exit when main program exits
        self.thread.start()

    def stop(self):
        """Stop the rendering thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)  # Wait for thread to finish with timeout

    def queue_render(self, iteration):
        """
        Queue a render operation if it's time to render.

        Args:
            iteration: Current training iteration
        """
        if iteration % self.render_interval == 0:
            # Just queue the iteration number, actual model state will be grabbed at render time
            self.queue.put(iteration)

    def _render_loop(self):
        """Main loop for the rendering thread."""
        while self.running:
            try:
                # Get next item from queue with timeout to allow checking running flag
                iteration = self.queue.get(timeout=0.5)

                with torch.no_grad():
                    # Perform the actual rendering
                    start_time = time.time()
                    self.model.pv_render()
                    render_time = time.time() - start_time

                    # Optional: print rendering stats occasionally
                    if iteration % (self.render_interval * 20) == 0:
                        print(
                            f"[Render Thread] Frame at iteration {iteration}, render time: {render_time:.4f}s"
                        )

                    self.queue.task_done()

            except queue.Empty:
                # Queue is empty, just continue the loop
                continue
            except Exception as e:
                print(f"[Render Thread] Error: {e}")


# Add this method to the SpherePacker class
def initialize_render_thread(self, render_interval=5):
    """
    Initialize and start a thread for rendering.

    Args:
        render_interval: How often to render frames (in iterations)
    """
    self.render_thread = RenderThread(self, render_interval)
    self.render_thread.start()


# Add this method to the SpherePacker class
def stop_render_thread(self):
    """Stop the rendering thread if it exists."""
    if hasattr(self, "render_thread"):
        self.render_thread.stop()
