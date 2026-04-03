"""add ivfflat profiles

Revision ID: 0002_add_ivfflat_profiles
Revises: 0001_init_pgvector_schema
Create Date: 2026-03-10

"""

from alembic import op


revision = "0002_add_ivfflat_profiles"
down_revision = "0001_init_pgvector_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding_768_ivfflat
        ON rag_chunks USING ivfflat ((embedding::vector(768)) vector_cosine_ops)
        WITH (lists = 100)
        WHERE vector_dims(embedding) = 768
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding_1536_ivfflat
        ON rag_chunks USING ivfflat ((embedding::vector(1536)) vector_cosine_ops)
        WITH (lists = 100)
        WHERE vector_dims(embedding) = 1536
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_rag_chunks_embedding_1536_ivfflat")
    op.execute("DROP INDEX IF EXISTS idx_rag_chunks_embedding_768_ivfflat")
