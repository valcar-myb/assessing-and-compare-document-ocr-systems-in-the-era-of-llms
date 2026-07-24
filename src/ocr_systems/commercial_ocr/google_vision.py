import json
from typing import Dict, Any

from google.cloud import vision
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToJson

from ..base import OCRSystem


class GoogleVisionOCR(OCRSystem):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)

        if 'type' in self.config and 'project_id' in self.config:
            credentials = service_account.Credentials.from_service_account_info({
                'type': self.config.get('type'),
                'project_id': self.config.get('project_id'),
                'private_key_id': self.config.get('private_key_id'),
                'private_key': self.config.get('private_key', '').replace('\\n', '\n'),
                'client_email': self.config.get('client_email'),
                'client_id': str(self.config.get('client_id')),
                'auth_uri': self.config.get('auth_uri'),
                'token_uri': self.config.get('token_uri'),
                'auth_provider_x509_cert_url': self.config.get('auth_provider_x509_cert_url'),
                'client_x509_cert_url': self.config.get('client_x509_cert_url'),
                'universe_domain': self.config.get('universe_domain', 'googleapis.com'),
            })
            self.model = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            self.model = vision.ImageAnnotatorClient()

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        with open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = self.model.text_detection(image=image)
        return json.loads(MessageToJson(response._pb))
