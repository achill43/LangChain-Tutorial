from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

DOCS_DATABASE_URL = "sqlite+aiosqlite:///./docs.db"

async_engine = create_async_engine(
    DOCS_DATABASE_URL,
    echo=True, # Good for debugging SQL queries
    pool_size=10, # Adjust pool size as needed
    max_overflow=20 # Adjust max_overflow as needed
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession, # Use AsyncSession for async operations
    expire_on_commit=False # Important for async: objects remain valid after commit
)

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
