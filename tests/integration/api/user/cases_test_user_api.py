import datetime
from http import HTTPStatus

from src.config import config
from src.interfaces.api.v1.schemas import (
    UserCreate,
    UserRead,
    UserUpdate,
    InferenceCreate,
    IDMixin,
)

from ..utils import APICase

now = datetime.datetime.now()

test_inference = InferenceCreate(
    user_id=2,
    celebrities=[],
    attractiveness=0.5,
    timestamp=now,
)
test_user_create = UserCreate(name="test", password="test")
test_id_read = IDMixin(id=2)
test_user_read = UserRead(id=2, name="test")
test_user_update = UserUpdate(name="test2")

admin_login = (config.auth.admin_name, config.auth.admin_password)
user_login = ("test", "test")


class CasesUserCreate:
    def case_create_success(self):
        return APICase(
            login=admin_login,
            endpoint_to_test="/users/",
            method="post",
            inp_body=test_user_create.model_dump(),
            expected_status=HTTPStatus.CREATED,
            expected_body=test_user_read.model_dump(),
        )

    def case_create_already_exists(self):
        return APICase(
            login=admin_login,
            endpoint_to_test="/users/",
            method="post",
            inp_body=test_user_create.model_dump(),
            expected_status=HTTPStatus.CONFLICT,
        )


class CasesUserGet:
    def case_get_users_success(self):
        return APICase(
            login=admin_login,
            endpoint_to_test="/users/",
            method="get",
            expected_status=HTTPStatus.OK,
            expected_body=[test_user_read.model_dump()],
        )

    def case_get_me_success(self):
        return APICase(
            login=user_login,
            endpoint_to_test="/users/me/",
            method="get",
            expected_status=HTTPStatus.OK,
            expected_body=test_user_read.model_dump(),
        )

    def case_get_me_unauthorized(self):
        return APICase(
            login=("", ""),
            endpoint_to_test="/users/me/",
            method="get",
            expected_status=HTTPStatus.UNAUTHORIZED,
        )

    def case_get_user_success(self):
        return APICase(
            login=admin_login,
            endpoint_to_test=f"/users/{test_user_read.id}/",
            method="get",
            expected_status=HTTPStatus.OK,
            expected_body=test_user_read.model_dump(),
        )

    def case_get_user_not_found(self):
        return APICase(
            login=admin_login,
            endpoint_to_test=f"/users/{test_user_read.id + 1}/",
            method="get",
            expected_status=HTTPStatus.NOT_FOUND,
        )

    def case_get_user_unauthorized(self):
        return APICase(
            login=("", ""),
            endpoint_to_test=f"/users/{test_user_read.id}/",
            method="get",
            expected_status=HTTPStatus.UNAUTHORIZED,
        )


class CasesUpdateUser:
    def case_update_user_success(self):
        return APICase(
            login=admin_login,
            endpoint_to_test=f"/users/{test_user_read.id}/",
            method="patch",
            inp_body=test_user_update.model_dump(),
            expected_status=HTTPStatus.OK,
            expected_body=test_user_read.model_dump(),
        )

    def case_update_user_not_found(self):
        return APICase(
            login=admin_login,
            endpoint_to_test=f"/users/{test_user_read.id + 1}/",
            method="patch",
            inp_body=test_user_update.model_dump(),
            expected_status=HTTPStatus.NOT_FOUND,
        )

    def case_update_user_unauthorized(self):
        return APICase(
            login=("", ""),
            endpoint_to_test=f"/users/{test_user_read.id}/",
            method="patch",
            inp_body=test_user_update.model_dump(),
            expected_status=HTTPStatus.UNAUTHORIZED,
        )


class CasesDeleteUser:
    def case_delete_user_success(self):
        return APICase(
            login=admin_login,
            endpoint_to_test=f"/users/{test_user_read.id}/",
            method="delete",
            expected_status=HTTPStatus.OK,
        )

    def case_delete_user_not_found(self):
        return APICase(
            login=admin_login,
            endpoint_to_test=f"/users/{test_user_read.id + 1}/",
            method="delete",
            expected_status=HTTPStatus.NOT_FOUND,
        )

    def case_delete_user_unauthorized(self):
        return APICase(
            login=("", ""),
            endpoint_to_test=f"/users/{test_user_read.id}/",
            method="delete",
            expected_status=HTTPStatus.UNAUTHORIZED,
        )
