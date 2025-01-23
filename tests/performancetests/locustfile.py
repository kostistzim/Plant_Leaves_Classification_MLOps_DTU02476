from locust import HttpUser, between, task

from tests import _PATH_INTEGRATION_DATA
import random


class MyUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    wait_time = between(1, 2)

    @task
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the FastAPI app."""
        response = self.client.get("/")
        print(f"response root status code: {response.status_code}")

    @task(3)
    def test_predict_endpoint(self) -> None:
        """A task that simulates sending image files to the /predict/ endpoint."""
        image_file_paths = [
            _PATH_INTEGRATION_DATA / 'diseased.JPG',
            _PATH_INTEGRATION_DATA / 'healthy.JPG'
        ]

        image_file_path = random.choice(image_file_paths)

        with open(image_file_path, 'rb') as image_file:
            image_data = image_file.read()

        file_name = image_file_path.name
        files = {"data": (file_name, image_data, "image/jpeg")}

        response = self.client.post("/predict/", files=files)

        print(f"predict_endpoint status code: {response.status_code}")
        response_json = response.json()
