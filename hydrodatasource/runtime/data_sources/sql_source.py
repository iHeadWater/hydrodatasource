"""
SQL database data source for runtime data loading.

Supports various SQL databases including PostgreSQL, MySQL, SQLite, and SQL Server.
Optimized for real-time queries with connection pooling and query optimization.

Author: Wenyu Ouyang
Date: 2025-01-22
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from urllib.parse import urlparse

from .base_source import BaseDataSource

logger = logging.getLogger(__name__)

# Optional SQL dependencies
try:
    import sqlalchemy
    from sqlalchemy import create_engine, text, MetaData, Table
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import QueuePool

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    sqlalchemy = None
    SQLALCHEMY_AVAILABLE = False


class SQLDataSource(BaseDataSource):
    """
    SQL database data source with connection pooling and query optimization.

    Supports:
    - PostgreSQL, MySQL, SQLite, SQL Server
    - Connection pooling for performance
    - Flexible table schemas
    - Automatic query optimization
    - Custom SQL queries
    """

    def __init__(self, source_config: Dict[str, Any]):
        """
        Initialize SQL data source.

        Parameters
        ----------
        source_config : Dict[str, Any]
            Configuration with keys:
            - connection_string: str, SQL connection string
            - table_name: str, name of the data table
            - time_column: str, name of time column (default: 'time')
            - basin_column: str, name of basin ID column (default: 'basin_id')
            - schema: str, database schema (optional)
            - pool_size: int, connection pool size (default: 5)
            - max_overflow: int, max pool overflow (default: 10)
            - custom_query: str, custom SQL query template (optional)
            - **kwargs: Additional SQLAlchemy parameters
        """
        super().__init__(source_config)

        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy is required for SQL data source. Install with: pip install sqlalchemy"
            )

        self.connection_string = source_config["connection_string"]
        self.table_name = source_config["table_name"]
        self.time_column = source_config.get("time_column", "time")
        self.basin_column = source_config.get("basin_column", "basin_id")
        self.schema = source_config.get("schema", None)
        self.custom_query = source_config.get("custom_query", None)

        # Connection pool settings
        pool_size = source_config.get("pool_size", 5)
        max_overflow = source_config.get("max_overflow", 10)

        # Additional SQLAlchemy parameters
        self.engine_kwargs = {
            k: v
            for k, v in source_config.items()
            if k
            not in [
                "connection_string",
                "table_name",
                "time_column",
                "basin_column",
                "schema",
                "pool_size",
                "max_overflow",
                "custom_query",
                "cache_enabled",
            ]
        }

        # Create SQLAlchemy engine with connection pooling
        self.engine = create_engine(
            self.connection_string,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # Validate connections
            **self.engine_kwargs,
        )

        # Initialize metadata
        self.metadata = MetaData()
        self._table_info = None
        self._column_cache = {}

        logger.info(f"Initialized SQL data source: {self._get_db_type()}")

    def _get_db_type(self) -> str:
        """Get database type from connection string."""
        parsed = urlparse(self.connection_string)
        return parsed.scheme.split("+")[0] if "+" in parsed.scheme else parsed.scheme

    def connect(self):
        """Establish database connection and load table metadata."""
        try:
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            # Load table metadata
            self.metadata.reflect(bind=self.engine, schema=self.schema)

            # Get table info
            table_key = (
                f"{self.schema}.{self.table_name}" if self.schema else self.table_name
            )
            if table_key in self.metadata.tables:
                self._table_info = self.metadata.tables[table_key]
                logger.info(f"Connected to table: {table_key}")
            else:
                available_tables = list(self.metadata.tables.keys())
                raise ValueError(
                    f"Table {table_key} not found. Available tables: {available_tables}"
                )

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _build_query(
        self,
        variables: List[str],
        basin_ids: List[str],
        time_range: Tuple[Union[str, datetime], Union[str, datetime]],
        limit: Optional[int] = None,
    ) -> str:
        """Build optimized SQL query for data loading."""

        if self.custom_query:
            # Use custom query template
            query = self.custom_query.format(
                variables=", ".join(variables),
                basin_ids="', '".join(basin_ids),
                start_time=time_range[0],
                end_time=time_range[1],
                table_name=(
                    f"{self.schema}.{self.table_name}"
                    if self.schema
                    else self.table_name
                ),
            )
        else:
            # Build standard query
            table_name = (
                f"{self.schema}.{self.table_name}" if self.schema else self.table_name
            )

            # Select columns
            columns = [self.time_column, self.basin_column] + variables
            select_clause = ", ".join(f'"{col}"' for col in columns)

            # Build WHERE clause
            where_conditions = []

            # Time range filter
            where_conditions.append(f'"{self.time_column}" >= %s')
            where_conditions.append(f'"{self.time_column}" <= %s')

            # Basin filter
            basin_placeholders = ", ".join(["%s"] * len(basin_ids))
            where_conditions.append(f'"{self.basin_column}" IN ({basin_placeholders})')

            where_clause = " AND ".join(where_conditions)

            # Build complete query
            query = f"""
            SELECT {select_clause}
            FROM {table_name}
            WHERE {where_clause}
            ORDER BY "{self.time_column}", "{self.basin_column}"
            """

            if limit:
                query += f" LIMIT {limit}"

        return query.strip()

    def _execute_query(
        self,
        query: str,
        variables: List[str],
        basin_ids: List[str],
        time_range: Tuple[Union[str, datetime], Union[str, datetime]],
    ) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""

        # Prepare parameters
        if self.custom_query:
            # For custom queries, parameters are already embedded
            params = []
        else:
            # For standard queries, use parameterized query
            params = [time_range[0], time_range[1]] + basin_ids

        try:
            with self.engine.connect() as conn:
                if params:
                    df = pd.read_sql_query(query, conn, params=params)
                else:
                    df = pd.read_sql_query(query, conn)

            # Parse time column
            if self.time_column in df.columns:
                df[self.time_column] = pd.to_datetime(df[self.time_column])

            # Set MultiIndex
            if self.time_column in df.columns and self.basin_column in df.columns:
                df = df.set_index([self.time_column, self.basin_column])

            return df.sort_index()

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            raise

    def load_variables(
        self,
        variables: List[str],
        basin_ids: List[str],
        time_range: Tuple[Union[str, datetime], Union[str, datetime]],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load specified variables for given basins and time range.

        Parameters
        ----------
        variables : List[str]
            Variable names to load
        basin_ids : List[str]
            Basin IDs to load
        time_range : Tuple[Union[str, datetime], Union[str, datetime]]
            (start_time, end_time) tuple
        **kwargs
            Additional parameters:
            - limit: int, maximum number of rows to return
            - custom_query: str, override default query

        Returns
        -------
        pd.DataFrame
            DataFrame with MultiIndex (time, basin) and variables as columns
        """
        # Convert time range to datetime strings for SQL
        start_time = (
            pd.to_datetime(time_range[0])
            if isinstance(time_range[0], str)
            else time_range[0]
        )
        end_time = (
            pd.to_datetime(time_range[1])
            if isinstance(time_range[1], str)
            else time_range[1]
        )

        # Build and execute query
        query = self._build_query(
            variables, basin_ids, (start_time, end_time), kwargs.get("limit")
        )
        df = self._execute_query(query, variables, basin_ids, (start_time, end_time))

        # Select only requested variables
        available_vars = [v for v in variables if v in df.columns]
        if not available_vars:
            raise ValueError(
                f"None of the requested variables {variables} found in data"
            )

        missing_vars = [v for v in variables if v not in available_vars]
        if missing_vars:
            logger.warning(f"Variables not found: {missing_vars}")

        return df[available_vars].copy()

    def get_available_variables(
        self, basin_ids: Optional[List[str]] = None
    ) -> List[str]:
        """Get list of available variables from database."""
        if not self._table_info:
            self.connect()

        # Get column names excluding time and basin columns
        exclude_cols = {self.time_column, self.basin_column}
        variables = [
            col.name for col in self._table_info.columns if col.name not in exclude_cols
        ]

        return variables

    def get_available_basins(self) -> List[str]:
        """Get list of available basin IDs from database."""
        query = f'SELECT DISTINCT "{self.basin_column}" FROM '
        if self.schema:
            query += f'"{self.schema}"."{self.table_name}"'
        else:
            query += f'"{self.table_name}"'

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(query, conn)
                return df[self.basin_column].tolist()
        except Exception as e:
            logger.error(f"Failed to get available basins: {e}")
            return []

    def get_time_range(
        self, basin_ids: Optional[List[str]] = None
    ) -> Tuple[datetime, datetime]:
        """Get available time range from database."""
        query = f'SELECT MIN("{self.time_column}") as min_time, MAX("{self.time_column}") as max_time FROM '
        if self.schema:
            query += f'"{self.schema}"."{self.table_name}"'
        else:
            query += f'"{self.table_name}"'

        if basin_ids:
            basin_placeholders = ", ".join([f"'{bid}'" for bid in basin_ids])
            query += f' WHERE "{self.basin_column}" IN ({basin_placeholders})'

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(query, conn)
                min_time = pd.to_datetime(df["min_time"].iloc[0])
                max_time = pd.to_datetime(df["max_time"].iloc[0])
                return min_time, max_time
        except Exception as e:
            logger.error(f"Failed to get time range: {e}")
            raise

    def execute_custom_query(
        self, query: str, params: Optional[List] = None
    ) -> pd.DataFrame:
        """
        Execute a custom SQL query.

        Parameters
        ----------
        query : str
            SQL query string
        params : Optional[List]
            Query parameters

        Returns
        -------
        pd.DataFrame
            Query results
        """
        try:
            with self.engine.connect() as conn:
                if params:
                    df = pd.read_sql_query(query, conn, params=params)
                else:
                    df = pd.read_sql_query(query, conn)
            return df
        except Exception as e:
            logger.error(f"Custom query execution failed: {e}")
            raise

    def disconnect(self):
        """Close database connections."""
        if hasattr(self, "engine") and self.engine:
            self.engine.dispose()

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the SQL data source."""
        metadata = super().get_metadata()

        try:
            if not self._table_info:
                self.connect()

            metadata.update(
                {
                    "database_type": self._get_db_type(),
                    "table_name": self.table_name,
                    "schema": self.schema,
                    "time_column": self.time_column,
                    "basin_column": self.basin_column,
                    "total_columns": (
                        len(self._table_info.columns) if self._table_info else 0
                    ),
                    "connection_pool_size": (
                        self.engine.pool.size()
                        if hasattr(self.engine.pool, "size")
                        else "unknown"
                    ),
                }
            )

            # Get table statistics if possible
            try:
                stats_query = f"SELECT COUNT(*) as total_rows FROM "
                if self.schema:
                    stats_query += f'"{self.schema}"."{self.table_name}"'
                else:
                    stats_query += f'"{self.table_name}"'

                with self.engine.connect() as conn:
                    stats_df = pd.read_sql_query(stats_query, conn)
                    metadata["total_rows"] = int(stats_df["total_rows"].iloc[0])

            except Exception:
                pass  # Skip statistics if query fails

        except Exception as e:
            metadata["metadata_error"] = str(e)

        return metadata
