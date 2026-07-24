import json
from typing import Dict, Any

from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToJson

from ..base import OCRSystem


class GoogleDocumentOCR(OCRSystem):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        location = self.config.get('location', 'us')
        client_options = ClientOptions(
            api_endpoint=f"{location}-documentai.googleapis.com"
        )

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
            })
            self.model = documentai.DocumentProcessorServiceClient(
                client_options=client_options,
                credentials=credentials,
            )
        else:
            self.model = documentai.DocumentProcessorServiceClient(
                client_options=client_options,
            )

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        processor_version_id = self.config.get('processor_version_id')
        if processor_version_id:
            name = self.model.processor_version_path(
                project=self.config.get('project_id'),
                location=self.config.get('location', 'us'),
                processor=self.config.get('processor_id'),
                processor_version=processor_version_id,
            )
        else:
            name = self.model.processor_path(
                project=self.config.get('project_id'),
                location=self.config.get('location', 'us'),
                processor=self.config.get('processor_id'),
            )

        with open(image_path, 'rb') as image:
            image_content = image.read()

        img_type = image_path.split('.')[-1].lower()
        mime_type_map = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp',
        }
        mime_type = mime_type_map.get(img_type, 'image/jpeg')

        raw_document = documentai.RawDocument(
            content=image_content,
            mime_type=mime_type,
        )

        process_options = documentai.ProcessOptions(
            individual_page_selector=documentai.ProcessOptions.IndividualPageSelector(
                pages=[1]
            )
        )

        result = self.model.process_document(
            request=documentai.ProcessRequest(
                name=name,
                raw_document=raw_document,
                field_mask=self.config.get('field_mask'),
                process_options=process_options,
            )
        )

        json_result = json.loads(MessageToJson(result._pb))

        # Drop embedded page images to keep the raw output compact.
        if 'document' in json_result and 'pages' in json_result['document']:
            for page in json_result['document']['pages']:
                if 'image' in page:
                    del page['image']

        return json_result
