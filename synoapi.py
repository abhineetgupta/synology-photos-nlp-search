import json
import logging
import os
from getpass import getpass

import requests
import urllib3

urllib3.disable_warnings()

# create logger
logger = logging.getLogger(__name__)


class SynoAPI:
    def __init__(
        self,
        hostname: str,
        api_path: str,
        username: str = None,
        password: str = None,
        mfa: bool = False,
        api_trials: int = 5,
    ):
        self.url = f"https://{hostname}/{api_path}"
        self.sid = None
        self.api_trials = api_trials
        if username is not None and password is not None:
            self.sid = self._login(username=username, password=password, mfa=mfa)

    def post_api(self, api: str, method: str, version: int = 1, **kwargs):
        data = kwargs
        data["api"] = api
        data["method"] = method
        data["version"] = version
        trials = max(1, int(self.api_trials))
        for trial in range(trials):
            try:
                response = requests.post(self.url, verify=False, data=data)
                response.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                message = f"Request failed for {response.request.url}: {e}"
                logger.error(message)
                if trial >= (trials - 1):
                    raise e
                else:
                    logger.debug(f"Retrying API call for {api}-{method}...")
        return response

    def _login(
        self,
        username: str = None,
        password: str = None,
        mfa: bool = True,
        password_trials: int = 1,
        **kwargs,
    ):
        logger.debug("Attempting login...")
        if username is None:
            if "SYNO_USERNAME" not in os.environ:
                message = "Username missing for login. Provide to function call or add in environment variable SYNO_USERNAME."
                logger.error(message)
                raise ValueError(message)
            username = os.environ["SYNO_USERNAME"]

        trials = max(1, int(password_trials))
        for trial in range(trials):
            if trial > 0:
                logger.info("Login unsuccessful. Retrying...")
                password = None
            if password is None and (
                ("SYNO_PASSWORD" not in os.environ) or (not os.environ["SYNO_PASSWORD"])
            ):
                password = getpass(
                    prompt=f"Enter password for Synology user `{username}` : "
                )
            elif password is None:
                password = os.environ["SYNO_PASSWORD"]
            data = dict(account=username, passwd=password)
            if mfa or trial or ("SYNO_2FA" in os.environ and os.environ["SYNO_2FA"]):
                otp = input("Enter 2FA code. Leave empty if 2FA is not activated : ")
                if otp:
                    data["otp_code"] = otp
                    logger.debug("Login authentication with 2FA code.")

            response = self.post_api("SYNO.API.Auth", "login", "3", **data, **kwargs)

            response_json = response.json()
            if response_json.get("success", False):
                break
        if not (response_json.get("success", False)):
            message = (
                f"Login request failed with error {response_json['error']['code']}."
            )
            logger.error(message)
            raise requests.exceptions.HTTPError(message)
        logger.info("Login successful.")
        return response_json["data"]["sid"]

    def _is_sid_none(self, sid: str = None):
        if sid is None:
            logger.info("No login sid is provided. No action taken.")
            return False
        return True

    def login(
        self,
        username: str = None,
        password: str = None,
        mfa: bool = False,
        password_trials: int = 1,
        **kwargs,
    ):
        logger.debug("Attempting login...")
        self.sid = self._login(
            username=username,
            password=password,
            mfa=mfa,
            password_trials=password_trials,
            **kwargs,
        )
        logger.info("Logged in.")

    def logout(self, sid: str = None, **kwargs):
        logger.debug("Attempting logout...")
        if not self._is_sid_none(self.sid):
            return
        response = self.post_api(
            "SYNO.API.Auth", "logout", "3", _sid=self.sid, **kwargs
        )
        response_json = response.json()
        if not (response_json.get("success", False)):
            message = (
                f"Logout request failed with error {response_json['error']['code']}."
            )
            logger.error(message)
            raise requests.exceptions.HTTPError(message)
        self.sid = None
        logger.info("Logged out.")


class SynoPhotosSharedAPI(SynoAPI):
    def __init__(
        self,
        hostname: str,
        username: str = None,
        password: str = None,
        mfa: bool = False,
    ):
        api_path = "photo/webapi/entry.cgi"
        super().__init__(
            hostname=hostname,
            api_path=api_path,
            username=username,
            password=password,
            mfa=mfa,
        )
        self._PAGINATION_SIZE = 500

    def list_param(self, x):
        return json.dumps(x if type(x) is list else [x])

    def reindex(self, **kwargs):
        logger.debug("Attempting reindexing...")
        if not self._is_sid_none(self.sid):
            return
        response = self.post_api(
            "SYNO.FotoTeam.Index", "reindex", "1", _sid=self.sid, **kwargs
        )
        response_json = response.json()
        if not (response_json.get("success", False)):
            message = (
                f"Reindex request failed with error {response_json['error']['code']}."
            )
            logger.error(message)
            raise requests.exceptions.HTTPError(message)
        logger.debug("Reindexing complete.")

    def get_image_count(self, **kwargs):
        logger.debug("Getting image count...")
        if not self._is_sid_none(self.sid):
            return None
        response = self.post_api(
            "SYNO.FotoTeam.Browse.Item", "count", "1", _sid=self.sid, **kwargs
        )
        response_json = response.json()

        if not (response_json.get("success", False)):
            message = f"Image count request failed with error {response_json['error']['code']}."
            logger.error(message)
            raise requests.exceptions.HTTPError(message)
        logger.debug("Got image count successfully.")
        return response_json["data"]["count"]

    def get_folder_count(self, folder_id: int = None, **kwargs):
        logger.debug(f"Getting folder count for folder_id={folder_id}...")
        if not self._is_sid_none(self.sid):
            return None
        response = self.post_api(
            "SYNO.FotoTeam.Browse.Folder",
            "count",
            "1",
            _sid=self.sid,
            id=folder_id,
            **kwargs,
        )
        response_json = response.json()
        if not (response_json.get("success", False)):
            message = f"Folder count request failed with error {response_json['error']['code']}."
            logger.error(message)
            raise requests.exceptions.HTTPError(message)
        logger.debug(f"Got folder count for folder_id={folder_id} successfully.")
        return response_json["data"]["count"]

    def get_image_list(self, limit: int = None, **kwargs):
        logger.debug("Getting image list...")
        if not self._is_sid_none(self.sid):
            return None
        if limit is None:
            limit = self.get_image_count()
        result = []
        while len(result) < limit:
            curr_batch = min(self._PAGINATION_SIZE, limit - len(result))
            response = self.post_api(
                "SYNO.FotoTeam.Browse.Item",
                "list",
                "1",
                _sid=self.sid,
                offset=len(result),
                limit=curr_batch,
                **kwargs,
            )
            response_json = response.json()
            if not (response_json.get("success", False)):
                message = f"Image list request failed with error {response_json['error']['code']}."
                logger.error(message)
                raise requests.exceptions.HTTPError(message)
            result.extend(response_json["data"]["list"])
        logger.debug("Got image list successfully.")
        return result

    def get_subfolders(self, limit: int = None, folder_id: int = None, **kwargs):
        logger.debug(f"Getting subfolders for folder_id={folder_id}...")
        if not self._is_sid_none(self.sid):
            return None
        if limit is None:
            limit = self.get_folder_count(folder_id)
        while len(result) < limit:
            curr_batch = min(self._PAGINATION_SIZE, limit - len(result))
            response = self.post_api(
                "SYNO.FotoTeam.Browse.Folder",
                "list",
                "1",
                _sid=self.sid,
                offset=len(result),
                limit=curr_batch,
                id=folder_id,
                **kwargs,
            )
            response_json = response.json()
            if not (response_json.get("success", False)):
                message = f"Folder list request failed with error {response_json['error']['code']}."
                logger.error(message)
                raise requests.exceptions.HTTPError(message)
            result.extend(response_json["data"]["list"])
        logger.debug(f"Got subfolders for folder_id={folder_id} successfully.")
        return result

    def get_folder_parents(self, limit: int = 0, folder_id: int = None, **kwargs):
        logger.debug(f"Getting parents for folder_id={folder_id}...")
        if not self._is_sid_none(self.sid):
            return None
        result = []
        while len(result) < limit:
            curr_batch = min(LIST_BATCH_SIZE, limit - len(result))
            response = self.post_api(
                "SYNO.FotoTeam.Browse.Folder",
                "list_parents",
                "1",
                _sid=self.sid,
                offset=len(result),
                limit=curr_batch,
                id=folder_id,
                **kwargs,
            )
            response_json = response.json()
            if not (response_json.get("success", False)):
                message = f"Folder parent list request failed with error {response_json['error']['code']}."
                logger.error(message)
                raise requests.exceptions.HTTPError(message)
            result.extend(response_json["data"]["list"])
        logger.debug(f"Got parents for folder_id={folder_id} successfully.")
        return result

    def create_tag(self, tag_name: str = "", **kwargs):
        logger.debug(f"Creating new tag={tag_name}...")
        if not self._is_sid_none(self.sid):
            return None
        response = self.post_api(
            "SYNO.FotoTeam.Browse.GeneralTag",
            "create",
            "1",
            _sid=self.sid,
            name=tag_name,
            **kwargs,
        )
        response_json = response.json()
        if not (response_json.get("success", False)):
            message = f"Creating tag request failed with error {response_json['error']['code']}."
            logger.error(message)
            raise requests.exceptions.HTTPError(message)
        logger.debug(f"Created new tag={tag_name}.")
        return response_json["data"]["tag"]

    def get_tag_count(self, **kwargs):
        logger.debug("Getting tag count...")
        if not self._is_sid_none(self.sid):
            return None
        response = self.post_api(
            "SYNO.FotoTeam.Browse.GeneralTag", "count", "1", _sid=self.sid, **kwargs
        )
        response_json = response.json()

        if not (response_json.get("success", False)):
            message = (
                f"Tag count request failed with error {response_json['error']['code']}."
            )
            logger.error(message)
            raise requests.exceptions.HTTPError(message)
        logger.debug("Got tag count successfully.")
        return response_json["data"]["count"]

    def get_tag_list(self, limit: int = None, **kwargs):
        logger.debug("Getting tag list...")
        if not self._is_sid_none(self.sid):
            return None
        if limit is None:
            limit = self.get_tag_count()
        result = []
        while len(result) < limit:
            curr_batch = min(self._PAGINATION_SIZE, limit - len(result))
            response = self.post_api(
                "SYNO.FotoTeam.Browse.GeneralTag",
                "list",
                "1",
                _sid=self.sid,
                offset=len(result),
                limit=curr_batch,
                **kwargs,
            )
            response_json = response.json()
            if not (response_json.get("success", False)):
                message = f"Tag list request failed with error {response_json['error']['code']}."
                logger.error(message)
                raise requests.exceptions.HTTPError(message)
            result.extend(response_json["data"]["list"])
        logger.debug("Got tag list successfully.")
        return result

    def add_tags(self, image_ids: list = [], tag_ids: list = [], **kwargs):
        logger.debug(f"Adding tag={tag_ids} to images={image_ids}...")
        if not self._is_sid_none(self.sid):
            return None
        response = self.post_api(
            "SYNO.FotoTeam.Browse.Item",
            "add_tag",
            "1",
            _sid=self.sid,
            id=self.list_param(image_ids),
            tag=self.list_param(tag_ids),
            **kwargs,
        )
        response_json = response.json()
        if not (response_json.get("success", False)):
            message = f"Adding tags to images request failed with error {response_json['error']['code']}."
            logger.error(message)
            raise requests.exceptions.HTTPError(message)
        logger.debug(f"Added tag={tag_ids} to images={image_ids}.")

    def remove_tags(self, image_ids: list = [], tag_ids: list = [], **kwargs):
        logger.debug(f"Removing tag={tag_ids} from images={image_ids}...")
        if not self._is_sid_none(self.sid):
            return None
        response = self.post_api(
            "SYNO.FotoTeam.Browse.Item",
            "remove_tag",
            "1",
            _sid=self.sid,
            id=self.list_param(image_ids),
            tag=self.list_param(tag_ids),
            **kwargs,
        )
        response_json = response.json()
        if not (response_json.get("success", False)):
            message = f"Removing tags from images request failed with error {response_json['error']['code']}."
            logger.error(message)
            raise requests.exceptions.HTTPError(message)
        logger.debug(f"Removed tag={tag_ids} from images={image_ids}.")
