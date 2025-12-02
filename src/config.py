from typing import Literal
import warnings

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import dotenv


class DB(BaseModel):
    prod_uri: str
    dev_uri: str
    connection_timeout: int
    pool_size: int
    pool_timeout: int
    uri: str = ""


class API(BaseModel):
    host: str
    port: int


class ML(BaseModel):
    mlflow_tracking_url: str
    remote_ip: str | None = None
    celebrities_dataset_dir: str = "datasets/open_famous_people_faces"


class Auth(BaseModel):
    secret_key: str
    algorithm: str
    access_token_expire_m: int
    salt: bytes
    admin_name: str
    admin_password: str


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env.example", ".env"],
        env_nested_delimiter="__",
        extra="ignore",
    )
    env: Literal["dev", "prod"]
    app_name: str = "app"
    prod: bool = False
    db: DB
    api: API
    ml: ML
    auth: Auth
    otlp_grpc_endpoint: str = "tempo:4317"

    def model_post_init(self, context):
        self.prod = self.env == "prod"
        self.db.uri = (
            self.prod
            and self.db.prod_uri.replace("postgresql://", "postgresql+asyncpg://")
            or self.db.dev_uri.replace("sqlite://", "sqlite+aiosqlite://")
        )


config = Config()

warnings.filterwarnings(
    "ignore", category=FutureWarning, message="pynvml package is deprecated"
)
dotenv.load_dotenv()
