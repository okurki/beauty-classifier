"""add_celebrity_feedback_table

Revision ID: 13d4d40b3276
Revises: 54afea78c6bb
Create Date: 2025-11-30 18:45:10.373544

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '13d4d40b3276'
down_revision: Union[str, Sequence[str], None] = '54afea78c6bb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'celebrity_feedbacks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('inference_id', sa.Integer(), nullable=False),
        sa.Column('celebrity_id', sa.Integer(), nullable=False),
        sa.Column('feedback_type', sa.String(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['celebrity_id'], ['celebrities.id'], ),
        sa.ForeignKeyConstraint(['inference_id'], ['inferences.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_celebrity_feedbacks_celebrity_id'), 'celebrity_feedbacks', ['celebrity_id'], unique=False)
    op.create_index(op.f('ix_celebrity_feedbacks_user_id'), 'celebrity_feedbacks', ['user_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_celebrity_feedbacks_user_id'), table_name='celebrity_feedbacks')
    op.drop_index(op.f('ix_celebrity_feedbacks_celebrity_id'), table_name='celebrity_feedbacks')
    op.drop_table('celebrity_feedbacks')
