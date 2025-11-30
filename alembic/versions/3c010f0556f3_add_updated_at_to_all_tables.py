"""add_updated_at_to_all_tables

Revision ID: 3c010f0556f3
Revises: 13d4d40b3276
Create Date: 2025-11-30 19:42:31.523621

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3c010f0556f3'
down_revision: Union[str, Sequence[str], None] = '13d4d40b3276'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add updated_at column to all tables that inherit from EntityBase
    tables = ['users', 'celebrities', 'inferences', 'celebrity_feedbacks']
    
    for table in tables:
        op.add_column(
            table,
            sa.Column(
                'updated_at',
                sa.DateTime(),
                nullable=False,
                server_default=sa.text('now()')
            )
        )


def downgrade() -> None:
    """Downgrade schema."""
    # Remove updated_at column from all tables
    tables = ['users', 'celebrities', 'inferences', 'celebrity_feedbacks']
    
    for table in tables:
        op.drop_column(table, 'updated_at')
