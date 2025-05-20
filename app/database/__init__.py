import logging
from enum import Enum
from peewee import (
    BigIntegerField,
    DateTimeField,
    CharField,
    TextField,
    FloatField,
    IntegerField,
    Model,
    CompositeKey # For LLM model
)
from playhouse.pool import PooledPostgresqlDatabase
import json # For JSONField a bit simpler than RAGflow's utils
import datetime
import time # For timestamp

from app import config # Use our own config

logger = logging.getLogger(__name__)

# --- Custom Field Types (simplified from RAGflow) ---
class JSONField(TextField): # Store JSON as text
    def db_value(self, value):
        if value is None:
            return None
        return json.dumps(value)

    def python_value(self, value):
        if value is None:
            return None
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return value # Or raise error, or return default

class ListField(JSONField):
    def python_value(self, value):
        val = super().python_value(value)
        return val if isinstance(val, list) else []


# --- Utility for timestamps ---
def current_timestamp_ms():
    return int(time.time() * 1000)

def timestamp_ms_to_datetime(ts_ms):
    if ts_ms is None:
        return None
    return datetime.datetime.fromtimestamp(ts_ms / 1000.0)

# --- Database Connection ---
db_instance = None

def get_db():
    global db_instance
    if db_instance is None:
        logger.info(f"Initializing PooledPostgresqlDatabase for '{config.DB_NAME}' on {config.DB_HOST}:{config.DB_PORT}")
        db_instance = PooledPostgresqlDatabase(
            config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT,
            max_connections=config.DB_MAX_CONNECTIONS,
            stale_timeout=300,  # Client-side timeout for connections
        )
        logger.info("PostgreSQL database connector initialized.")
    return db_instance

# --- Base Model ---
class BaseModel(Model):
    create_time = BigIntegerField(null=True, index=True, help_text="Timestamp in milliseconds")
    # create_date removed, can be derived from create_time if needed
    update_time = BigIntegerField(null=True, index=True, help_text="Timestamp in milliseconds")
    # update_date removed

    def to_dict(self):
        return self.__data__

    def save(self, *args, **kwargs):
        ts = current_timestamp_ms()
        if not self.create_time: # Set on first save
            self.create_time = ts
        self.update_time = ts
        return super().save(*args, **kwargs)

    # Class methods insert/update normalization removed for simplicity with Peewee's default save
    # Peewee handles created_at/updated_at with auto_now_add/auto_now if using DateTimeField directly
    # But since RAGflow used BigIntegerField, we stick to manual timestamping via save()

    class Meta:
        database = get_db() # Deferred initialization of database

# Export the database instance for potential direct use (e.g., creating tables)
DB = get_db()
