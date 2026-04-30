# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AWS HealthLake client for FHIR operations."""

# Standard library imports
# Third-party imports
import boto3
import httpx
import re

# Local imports
from . import __version__
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError
from loguru import logger
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, urljoin, urlsplit


# HealthLake API limits
MAX_SEARCH_COUNT = 100  # Maximum number of resources per search request
DATASTORE_ID_LENGTH = 32  # AWS HealthLake datastore ID length

# Security validators
# Opaque pagination token: conservative charset. Deliberately narrow to
# what HealthLake's `page` cursor format uses.
_PAGINATION_TOKEN_MAX_LEN = 2048
_PAGINATION_TOKEN_RE = re.compile(r'^[A-Za-z0-9+=_\-.]+$')

# FHIR R4 resource type and id formats.
# Resource type: PascalCase, letters/digits only, <=64 chars.
# Resource id: [A-Za-z0-9.-]{1,64}  (per FHIR R4 spec).
_FHIR_RESOURCE_TYPE_RE = re.compile(r'^[A-Z][A-Za-z0-9]{0,63}$')
_FHIR_RESOURCE_ID_RE = re.compile(r'^[A-Za-z0-9\-.]{1,64}$')


def validate_datastore_id(datastore_id: str) -> str:
    """Validate AWS HealthLake datastore ID format.

    Requires exactly DATASTORE_ID_LENGTH alphanumeric characters. Used to
    constrain the value before it is interpolated into FHIR endpoint URLs.
    """
    if (
        not datastore_id
        or not isinstance(datastore_id, str)
        or len(datastore_id) != DATASTORE_ID_LENGTH
        or not datastore_id.isalnum()
    ):
        raise ValueError(f'Datastore ID must be {DATASTORE_ID_LENGTH} alphanumeric characters')
    return datastore_id


def validate_pagination_token(next_token: Any) -> str:
    """Validate an opaque pagination token.

    The token is the server-emitted `page` query value from a prior
    HealthLake response. It must be a short, well-formed opaque string.
    Error messages do not include the input value.
    """
    if not isinstance(next_token, str) or not next_token:
        raise ValueError('Invalid pagination token')
    if len(next_token) > _PAGINATION_TOKEN_MAX_LEN:
        raise ValueError('Invalid pagination token')
    if not _PAGINATION_TOKEN_RE.match(next_token):
        raise ValueError('Invalid pagination token')
    return next_token


def validate_fhir_resource_type(resource_type: Any) -> str:
    """Validate a FHIR resource type (PascalCase, letters/digits only)."""
    if not isinstance(resource_type, str) or not _FHIR_RESOURCE_TYPE_RE.match(resource_type):
        raise ValueError('Invalid FHIR resource type')
    return resource_type


def validate_fhir_resource_id(resource_id: Any) -> str:
    """Validate a FHIR resource id per FHIR R4 ([A-Za-z0-9.-]{1,64})."""
    if not isinstance(resource_id, str) or not _FHIR_RESOURCE_ID_RE.match(resource_id):
        raise ValueError('Invalid FHIR resource id')
    return resource_id


class FHIRSearchError(Exception):
    """Exception raised for FHIR search parameter errors."""

    def __init__(self, message: str, invalid_params: Optional[List[str]] = None):
        """Initialize FHIR search error with message and optional invalid parameters."""
        self.invalid_params = invalid_params or []
        super().__init__(message)


class AWSAuth(httpx.Auth):
    """Custom AWS SigV4 authentication for httpx."""

    def __init__(
        self,
        credentials,
        region: str,
        service: str = 'healthlake',
        expected_host: Optional[str] = None,
    ):
        """Initialize AWS SigV4 authentication.

        If ``expected_host`` is provided, requests whose URL host does not
        match (case-insensitive) will be refused before any signing occurs.

        When ``expected_host`` is ``None``, behavior is unchanged (backward
        compatible for non-pagination call sites).
        """
        self.credentials = credentials
        self.region = region
        self.service = service
        self.expected_host = expected_host.lower() if expected_host else None

    def auth_flow(self, request):
        """Apply AWS SigV4 authentication to the request."""
        # Host allowlist check: refuse to sign for unexpected hosts.
        if self.expected_host is not None:
            request_host = (request.url.host or '').lower()
            if request_host != self.expected_host:
                # Log/exception messages do not include the URL or host.
                logger.warning('Refusing to sign request: unexpected host')
                raise ValueError('Refusing to sign request to unexpected host')

        # Preserve the original Content-Length if it exists
        original_content_length = request.headers.get('content-length')

        # Use minimal headers for signing - include Content-Length if present
        headers = {
            'Accept': 'application/fhir+json',
            'Content-Type': 'application/fhir+json',
            'Host': request.url.host,
        }

        # Add Content-Length to headers for signing if present
        if original_content_length:
            headers['Content-Length'] = original_content_length

        # For GET requests, no body
        body = None if request.method.upper() == 'GET' else request.content

        # Create AWS request for signing
        aws_request = AWSRequest(
            method=request.method, url=str(request.url), data=body, headers=headers
        )

        # Sign the request
        signer = SigV4Auth(self.credentials, self.service, self.region)
        signer.add_auth(aws_request)

        # Clear existing headers and set only the signed ones
        request.headers.clear()
        for key, value in aws_request.headers.items():
            request.headers[key] = value

        yield request


class HealthLakeClient:
    """Client for AWS HealthLake FHIR operations."""

    def __init__(self, region_name: Optional[str] = None):
        """Initialize the HealthLake client."""
        try:
            self.session = boto3.Session()
            self.healthlake_client = self.session.client(
                'healthlake',
                region_name=region_name,
                config=Config(user_agent_extra=f'awslabs/mcp/healthlake-mcp-server/{__version__}'),
            )
            self.region = region_name or self.session.region_name or 'us-east-1'

        except NoCredentialsError:
            logger.error('AWS credentials not found. Please configure your credentials.')
            raise

    async def list_datastores(self, filter_status: Optional[str] = None) -> Dict[str, Any]:
        """List HealthLake datastores."""
        try:
            kwargs = {}
            if filter_status:
                kwargs['Filter'] = {'DatastoreStatus': filter_status}

            response = self.healthlake_client.list_fhir_datastores(**kwargs)
            return response
        except ClientError as e:
            logger.error(f'Error listing datastores: {e}')
            raise

    async def get_datastore_details(self, datastore_id: str) -> Dict[str, Any]:
        """Get details of a specific datastore."""
        try:
            response = self.healthlake_client.describe_fhir_datastore(DatastoreId=datastore_id)
            return response
        except ClientError as e:
            logger.error(f'Error getting datastore details: {e}')
            raise

    def _get_fhir_endpoint(self, datastore_id: str) -> str:
        """Get the FHIR endpoint URL for a datastore."""
        return f'https://healthlake.{self.region}.amazonaws.com/datastore/{datastore_id}/r4/'

    def _build_search_request(
        self,
        base_url: str,
        resource_type: str,
        search_params: Optional[Dict[str, Any]] = None,
        include_params: Optional[List[str]] = None,
        revinclude_params: Optional[List[str]] = None,
        chained_params: Optional[Dict[str, str]] = None,
        count: int = 100,
        next_token: Optional[str] = None,
    ) -> Tuple[str, Dict[str, str]]:
        """Build search request with minimal processing."""
        # Handle pagination first
        if next_token:
            return next_token, {}

        # Build the search URL
        url = f'{base_url.rstrip("/")}/{resource_type}/_search'

        # Build form data with minimal processing
        form_data = {'_count': str(count)}

        # Add basic search parameters with proper encoding for FHIR modifiers
        if search_params:
            for key, value in search_params.items():
                # URL-encode colons in parameter names for FHIR modifiers
                encoded_key = key.replace(':', '%3A')
                if isinstance(value, list):
                    form_data[encoded_key] = ','.join(str(v) for v in value)
                else:
                    form_data[encoded_key] = str(value)

        # Add chained parameters with proper encoding for FHIR modifiers
        if chained_params:
            for key, value in chained_params.items():
                # URL-encode colons in parameter names for FHIR modifiers
                encoded_key = key.replace(':', '%3A')
                form_data[encoded_key] = str(value)

        # Add include parameters
        if include_params:
            form_data['_include'] = ','.join(include_params)

        # Add revinclude parameters
        if revinclude_params:
            form_data['_revinclude'] = ','.join(revinclude_params)

        return url, form_data

    def _validate_search_request(
        self,
        resource_type: str,
        search_params: Optional[Dict[str, Any]] = None,
        include_params: Optional[List[str]] = None,
        revinclude_params: Optional[List[str]] = None,
        chained_params: Optional[Dict[str, str]] = None,
        count: int = 100,
    ) -> List[str]:
        """Minimal validation - only catch obvious errors."""
        errors = []

        # Basic sanity checks only
        if not resource_type or not resource_type.strip():
            errors.append('Resource type is required')

        if count < 1 or count > 100:
            errors.append('Count must be between 1 and 100')

        # Basic format checks for include parameters
        if include_params:
            for param in include_params:
                if ':' not in param:
                    errors.append(
                        f"Invalid include format: '{param}'. Expected 'ResourceType:parameter'"
                    )

        if revinclude_params:
            for param in revinclude_params:
                if ':' not in param:
                    errors.append(
                        f"Invalid revinclude format: '{param}'. Expected 'ResourceType:parameter'"
                    )

        return errors

    def _extract_next_page_token(self, bundle: Dict[str, Any]) -> Optional[str]:
        """Extract an opaque ``page`` token from a Bundle's ``next`` link.

        Returns just the ``page`` query parameter value from HealthLake's
        ``link[rel=next]`` URL, or ``None`` if no next link exists. The URL
        itself is not surfaced to callers; the paginated URL is
        reconstructed server-side from trusted components on the next call.

        Assumptions (HealthLake-specific; may not hold for other FHIR R4
        servers):
          * The ``next`` link is a URL on the same HealthLake datastore.
          * The only caller-relevant continuation state is the ``page``
            query parameter; other query params (e.g. ``_count``) are
            either re-supplied by this client or safely defaulted.
          * HealthLake emits ``next`` URLs against ``/<ResourceType>`` for
            search and ``/Patient/<id>/$everything`` for $patient-everything.
        If HealthLake ever changes the continuation URL shape or introduces
        additional required query parameters, this extractor and the
        ``_build_*_pagination_url`` helpers must be updated together.
        """
        next_url = None
        for link in bundle.get('link', []) or []:
            if link.get('relation') == 'next':
                next_url = link.get('url', '')
                break

        if not next_url:
            return None

        try:
            link_qs = parse_qs(urlsplit(next_url).query)
        except Exception:
            logger.warning('Failed to parse pagination next link')
            return None

        page_values = link_qs.get('page')
        if not page_values:
            return None

        candidate = page_values[0]
        try:
            # Re-validate shape so we never return anything that would fail
            # validate_pagination_token on the next call. Also keeps the
            # payload format stable for MCP clients.
            return validate_pagination_token(candidate)
        except ValueError:
            logger.warning('Discarding malformed pagination page value from HealthLake')
            return None

    def _process_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Process FHIR Bundle response and extract pagination information.

        ``pagination.next_token`` is the opaque ``page`` value extracted
        from the Bundle's ``next`` link, or ``None``. It is NOT a URL.
        Callers pass it back as-is and the server reconstructs the full
        URL from trusted components.
        """
        result = {
            'resourceType': bundle.get('resourceType', 'Bundle'),
            'id': bundle.get('id'),
            'type': bundle.get('type', 'searchset'),
            'total': bundle.get('total'),
            'entry': bundle.get('entry', []),
            'link': bundle.get('link', []),
        }

        # Add total field if not present (some HealthLake responses may not include it)
        if 'total' not in result or result['total'] is None:
            result['total'] = len(result.get('entry', []))

        next_token = self._extract_next_page_token(bundle)
        result['pagination'] = {'has_next': next_token is not None, 'next_token': next_token}
        return result

    def _process_bundle_with_includes(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Process bundle and organize included resources.

        ``pagination.next_token`` semantics match ``_process_bundle``: opaque
        page value only, never a URL.
        """
        # Separate main results from included resources
        main_entries = []
        included_entries = []

        for entry in bundle.get('entry', []):
            search_mode = entry.get('search', {}).get('mode', 'match')
            if search_mode == 'match':
                main_entries.append(entry)
            elif search_mode == 'include':
                included_entries.append(entry)

        # Organize included resources by type and ID for easier access
        included_by_type: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for entry in included_entries:
            resource = entry.get('resource', {})
            resource_type = resource.get('resourceType')
            resource_id = resource.get('id')

            if resource_type and resource_id:
                if resource_type not in included_by_type:
                    included_by_type[resource_type] = {}
                included_by_type[resource_type][resource_id] = resource

        # Build response
        result = {
            'resourceType': bundle.get('resourceType', 'Bundle'),
            'id': bundle.get('id'),
            'type': bundle.get('type', 'searchset'),
            'total': bundle.get('total', len(main_entries)),  # Use main_entries count as fallback
            'entry': main_entries,
            'link': bundle.get('link', []),
        }

        # Add organized included resources
        if included_by_type:
            result['included'] = included_by_type

        next_token = self._extract_next_page_token(bundle)
        result['pagination'] = {'has_next': next_token is not None, 'next_token': next_token}

        return result

    def _create_helpful_error_message(self, error: Exception) -> str:
        """Create helpful error messages without over-engineering."""
        error_str = str(error)

        # Simple, actionable guidance
        if '400' in error_str:
            return (
                f'HealthLake rejected the search request: {error_str}\n\n'
                '💡 Common solutions:\n'
                '• Check parameter names and values\n'
                '• Try simpler search parameters\n'
                '• Verify resource type is correct\n'
                '• Some advanced FHIR features may not be supported'
            )
        elif 'validation' in error_str.lower():
            return (
                f'Search validation failed: {error_str}\n\n'
                '💡 Check your search parameters format and try again.'
            )
        else:
            return f'Search error: {error_str}'

    def _build_pagination_url(
        self, datastore_id: str, resource_type: str, count: int, next_token: str
    ) -> str:
        """Reconstruct a paginated search URL from trusted components.

        Scheme, host, port, and path are fixed by this client; only the
        ``page`` query parameter value comes from the caller, and must
        already have been validated by ``validate_pagination_token``.

        HealthLake emits the ``next`` link on the ``/{resource_type}``
        (GET) path, not on ``/{resource_type}/_search`` (POST). We
        reconstruct the GET path here to match.
        """
        endpoint = self._get_fhir_endpoint(datastore_id).rstrip('/')
        return f'{endpoint}/{resource_type}?_count={count}&page={quote(next_token, safe="")}'

    def _build_patient_everything_pagination_url(
        self, datastore_id: str, patient_id: str, count: int, next_token: str
    ) -> str:
        """Reconstruct a paginated $everything URL from trusted components."""
        endpoint = self._get_fhir_endpoint(datastore_id).rstrip('/')
        return (
            f'{endpoint}/Patient/{patient_id}/$everything'
            f'?_count={count}&page={quote(next_token, safe="")}'
        )

    async def patient_everything(
        self,
        datastore_id: str,
        patient_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        count: int = 100,
        next_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve all resources related to a specific patient via $patient-everything."""
        # Input validation runs before any I/O.
        validate_datastore_id(datastore_id)
        validate_fhir_resource_id(patient_id)
        if next_token is not None:
            validate_pagination_token(next_token)

        try:
            endpoint = self._get_fhir_endpoint(datastore_id)
            auth = self._get_aws_auth(expected_host=self._healthlake_host())

            # Ensure count is within valid range
            count = max(1, min(count, MAX_SEARCH_COUNT))

            # follow_redirects=False (explicit) -- httpx's default is False
            # today, but being explicit keeps behavior stable if that default
            # ever changes.
            async with httpx.AsyncClient(follow_redirects=False) as client:
                if next_token:
                    url = self._build_patient_everything_pagination_url(
                        datastore_id=datastore_id,
                        patient_id=patient_id,
                        count=count,
                        next_token=next_token,
                    )
                    response = await client.get(url, auth=auth)
                else:
                    # Build $patient-everything URL
                    url = urljoin(endpoint, f'Patient/{patient_id}/$everything')

                    # Build query parameters
                    params = {'_count': str(count)}
                    if start:
                        params['start'] = start
                    if end:
                        params['end'] = end

                    logger.debug(f'Query params: {params}')

                    response = await client.get(url, params=params, auth=auth)

                response.raise_for_status()
                fhir_bundle = response.json()

                # Process the response
                result = self._process_bundle(fhir_bundle)
                return result

        except Exception as e:
            logger.error(f'Error in patient everything operation: {e}')
            raise

    async def search_resources(
        self,
        datastore_id: str,
        resource_type: str,
        search_params: Optional[Dict[str, str]] = None,
        include_params: Optional[List[str]] = None,
        revinclude_params: Optional[List[str]] = None,
        chained_params: Optional[Dict[str, str]] = None,
        count: int = 100,
        next_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search for FHIR resources."""
        # Input validation runs before the try block so ValueError is not
        # rewrapped by _create_helpful_error_message.
        validate_datastore_id(datastore_id)
        validate_fhir_resource_type(resource_type)
        if next_token is not None:
            validate_pagination_token(next_token)

        try:
            endpoint = self._get_fhir_endpoint(datastore_id)
            auth = self._get_aws_auth(expected_host=self._healthlake_host())

            # Ensure count is within valid range
            count = max(1, min(count, MAX_SEARCH_COUNT))

            # Minimal validation
            validation_errors = self._validate_search_request(
                resource_type=resource_type,
                search_params=search_params,
                include_params=include_params,
                revinclude_params=revinclude_params,
                chained_params=chained_params,
                count=count,
            )

            if validation_errors:
                raise FHIRSearchError(f'Search validation failed: {"; ".join(validation_errors)}')

            # Build request (for non-paginated case)
            url, form_data = self._build_search_request(
                base_url=endpoint,
                resource_type=resource_type,
                search_params=search_params,
                include_params=include_params,
                revinclude_params=revinclude_params,
                chained_params=chained_params,
                count=count,
                next_token=None,  # Pagination URL is built separately below.
            )

            async with httpx.AsyncClient(follow_redirects=False) as client:
                if next_token:
                    pagination_url = self._build_pagination_url(
                        datastore_id=datastore_id,
                        resource_type=resource_type,
                        count=count,
                        next_token=next_token,
                    )
                    response = await client.get(pagination_url, auth=auth)
                else:
                    # Use POST for search
                    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

                    logger.debug(f'Search URL: {url}')
                    logger.debug(f'Form data: {form_data}')

                    response = await client.post(url, data=form_data, headers=headers, auth=auth)

                response.raise_for_status()
                fhir_bundle = response.json()

                # Process response with appropriate handling
                has_includes = bool(include_params or revinclude_params)
                if has_includes:
                    result = self._process_bundle_with_includes(fhir_bundle)
                else:
                    result = self._process_bundle(fhir_bundle)

                return result

        except FHIRSearchError:
            # Re-raise FHIR search errors as-is
            raise
        except Exception as e:
            logger.error(f'Error searching resources: {e}')
            # Provide helpful error message
            raise Exception(self._create_helpful_error_message(e))

    async def read_resource(
        self, datastore_id: str, resource_type: str, resource_id: str
    ) -> Dict[str, Any]:
        """Get a specific FHIR resource by ID."""
        validate_datastore_id(datastore_id)
        validate_fhir_resource_type(resource_type)
        validate_fhir_resource_id(resource_id)
        try:
            endpoint = self._get_fhir_endpoint(datastore_id)
            url = urljoin(endpoint, f'{resource_type}/{resource_id}')

            auth = self._get_aws_auth(expected_host=self._healthlake_host())

            async with httpx.AsyncClient(follow_redirects=False) as client:
                response = await client.get(url, auth=auth)
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f'Error getting resource: {e}')
            raise

    async def create_resource(
        self, datastore_id: str, resource_type: str, resource_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new FHIR resource."""
        validate_datastore_id(datastore_id)
        validate_fhir_resource_type(resource_type)
        try:
            endpoint = self._get_fhir_endpoint(datastore_id)
            url = urljoin(endpoint, resource_type)

            # Ensure resource has correct resourceType
            resource_data['resourceType'] = resource_type

            auth = self._get_aws_auth(expected_host=self._healthlake_host())

            async with httpx.AsyncClient(follow_redirects=False) as client:
                response = await client.post(url, json=resource_data, auth=auth)
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f'Error creating resource: {e}')
            raise

    async def update_resource(
        self,
        datastore_id: str,
        resource_type: str,
        resource_id: str,
        resource_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update an existing FHIR resource."""
        validate_datastore_id(datastore_id)
        validate_fhir_resource_type(resource_type)
        validate_fhir_resource_id(resource_id)
        try:
            endpoint = self._get_fhir_endpoint(datastore_id)
            url = urljoin(endpoint, f'{resource_type}/{resource_id}')

            # Ensure resource has correct resourceType and id
            resource_data['resourceType'] = resource_type
            resource_data['id'] = resource_id

            auth = self._get_aws_auth(expected_host=self._healthlake_host())

            async with httpx.AsyncClient(follow_redirects=False) as client:
                response = await client.put(url, json=resource_data, auth=auth)
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f'Error updating resource: {e}')
            raise

    async def delete_resource(
        self, datastore_id: str, resource_type: str, resource_id: str
    ) -> Dict[str, Any]:
        """Delete a FHIR resource."""
        validate_datastore_id(datastore_id)
        validate_fhir_resource_type(resource_type)
        validate_fhir_resource_id(resource_id)
        try:
            endpoint = self._get_fhir_endpoint(datastore_id)
            url = urljoin(endpoint, f'{resource_type}/{resource_id}')

            auth = self._get_aws_auth(expected_host=self._healthlake_host())

            async with httpx.AsyncClient(follow_redirects=False) as client:
                response = await client.delete(url, auth=auth)
                response.raise_for_status()
                return {'status': 'deleted', 'resourceType': resource_type, 'id': resource_id}

        except Exception as e:
            logger.error(f'Error deleting resource: {e}')
            raise

    async def start_import_job(
        self,
        datastore_id: str,
        input_data_config: Dict[str, Any],
        job_output_data_config: Dict[str, Any],
        data_access_role_arn: str,
        job_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start a FHIR import job."""
        try:
            # Validate required parameters
            if not input_data_config.get('s3_uri'):
                raise ValueError("input_data_config must contain 's3_uri'")

            if not job_output_data_config.get('s3_configuration', {}).get('s3_uri'):
                raise ValueError(
                    'job_output_data_config must contain s3_configuration with s3_uri'
                )

            # Transform input_data_config to match AWS API format
            input_config = {'S3Uri': input_data_config['s3_uri']}

            # Transform job_output_data_config to match AWS API format
            s3_config = job_output_data_config['s3_configuration']
            output_config = {'S3Configuration': {'S3Uri': s3_config['s3_uri']}}

            # Add KMS key if provided
            if s3_config.get('kms_key_id'):
                output_config['S3Configuration']['KmsKeyId'] = s3_config['kms_key_id']

            kwargs = {
                'DatastoreId': datastore_id,
                'InputDataConfig': input_config,
                'JobOutputDataConfig': output_config,
                'DataAccessRoleArn': data_access_role_arn,
            }

            if job_name:
                kwargs['JobName'] = job_name

            response = self.healthlake_client.start_fhir_import_job(**kwargs)
            return response

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))

            # Provide more specific error messages
            if error_code == 'ValidationException':
                logger.error(f'Validation error starting import job: {error_message}')
                raise ValueError(f'Invalid parameters: {error_message}')
            elif error_code == 'AccessDeniedException':
                logger.error(f'Access denied starting import job: {error_message}')
                raise PermissionError(f'Access denied: {error_message}')
            elif error_code == 'ResourceNotFoundException':
                logger.error(f'Resource not found starting import job: {error_message}')
                raise ValueError(f'Datastore not found: {error_message}')
            else:
                logger.error(f'Error starting import job: {error_message}')
                raise

    async def start_export_job(
        self,
        datastore_id: str,
        output_data_config: Dict[str, Any],
        data_access_role_arn: str,
        job_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start a FHIR export job."""
        try:
            kwargs = {
                'DatastoreId': datastore_id,
                'OutputDataConfig': output_data_config,
                'DataAccessRoleArn': data_access_role_arn,
            }
            if job_name:
                kwargs['JobName'] = job_name

            response = self.healthlake_client.start_fhir_export_job(**kwargs)
            return response
        except ClientError as e:
            logger.error(f'Error starting export job: {e}')
            raise

    async def list_jobs(
        self, datastore_id: str, job_status: Optional[str] = None, job_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """List FHIR import/export jobs."""
        try:
            if job_type == 'IMPORT':
                kwargs: Dict[str, Any] = {'DatastoreId': datastore_id}
                if job_status:
                    kwargs['JobStatus'] = job_status
                response = self.healthlake_client.list_fhir_import_jobs(**kwargs)
            elif job_type == 'EXPORT':
                kwargs: Dict[str, Any] = {'DatastoreId': datastore_id}
                if job_status:
                    kwargs['JobStatus'] = job_status
                response = self.healthlake_client.list_fhir_export_jobs(**kwargs)
            else:
                # List both import and export jobs
                import_jobs = self.healthlake_client.list_fhir_import_jobs(
                    DatastoreId=datastore_id
                )
                export_jobs = self.healthlake_client.list_fhir_export_jobs(
                    DatastoreId=datastore_id
                )
                response = {
                    'ImportJobs': import_jobs.get('ImportJobPropertiesList', []),
                    'ExportJobs': export_jobs.get('ExportJobPropertiesList', []),
                }
            return response
        except ClientError as e:
            logger.error(f'Error listing jobs: {e}')
            # Return error information instead of crashing
            return {'error': True, 'message': str(e), 'ImportJobs': [], 'ExportJobs': []}

    def _get_aws_auth(self, expected_host: Optional[str] = None):
        """Get AWS authentication for HTTP requests.

        If ``expected_host`` is provided, the returned auth will refuse to
        sign requests whose host does not match.
        """
        try:
            # Get AWS credentials from the session
            credentials = self.session.get_credentials()
            if not credentials:
                raise NoCredentialsError()

            # Create custom AWS authentication instance
            auth = AWSAuth(
                credentials=credentials,
                region=self.region,
                service='healthlake',
                expected_host=expected_host,
            )

            return auth

        except Exception as e:
            logger.error(f'Failed to get AWS authentication: {e}')
            raise

    def _healthlake_host(self) -> str:
        """Return the expected HealthLake hostname for this client's region."""
        return f'healthlake.{self.region}.amazonaws.com'
