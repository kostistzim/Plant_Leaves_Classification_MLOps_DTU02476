from fastapi.testclient import TestClient
from plant_leaves.api import app
from http import HTTPStatus

from tests import _PATH_TEST_DATA


client = TestClient(app)

def test_read_root():
    with TestClient(app) as client: # because we use lifespan
        response = client.get("/")
        assert response.json()["message"] == HTTPStatus.OK.phrase
        assert response.status_code == HTTPStatus.OK


def test_predict_endpoint():
    # Define the image file paths
    image_file_paths = [
        _PATH_TEST_DATA.joinpath("raw/plant-leaves-for-image-classification/Plants_2/valid/Alstonia Scholaris diseased (P2a)/diseased.JPG"),
        _PATH_TEST_DATA.joinpath("raw/plant-leaves-for-image-classification/Plants_2/valid/Alstonia Scholaris healthy (P2b)/healthy.JPG")
    ]
    for image_file_path in image_file_paths:
        with open(image_file_path, 'rb') as image_file:
            image_data = image_file.read()

        with TestClient(app) as client:  # because we use lifespan
            file_name = image_file_path.name
            file_label = file_name.split(".")[0]
            files = {"data": (file_name, image_data, "image/jpeg")}

            response = client.post("/predict/", files=files)

            response_json = response.json()

            assert response.status_code == HTTPStatus.OK
            assert response_json["status_code"] == HTTPStatus.OK
            assert response_json["image_label"] in ["healthy", "diseased"]
            print(f"Tested {file_name}: {response_json['image_label']}")

