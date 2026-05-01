# File: seed.py
import asyncio

from app.auth.security import get_password_hash
from app.config import get_settings
from app.db.models import Role, User
from app.db.session import create_engine, get_session_factory


async def seed():
    settings = get_settings()
    engine = create_engine(str(settings.postgres_dsn))
    session_factory = get_session_factory(engine)

    async with session_factory() as session:
        # 1. Створюємо ролі
        hr_role = Role(name="hr_manager", permissions=["view_unmasked", "upload_docs"])
        viewer_role = Role(name="viewer", permissions=["view_masked"])
        session.add_all([hr_role, viewer_role])
        await session.commit()

        # 2. Створюємо користувачів
        hr_user = User(
            email="hr@example.com",
            hashed_password=get_password_hash("12345"),
            role_name="hr_manager",
            department_id="hr_dept",
        )
        viewer_user = User(
            email="viewer@example.com",
            hashed_password=get_password_hash("12345"),
            role_name="viewer",
            department_id="hr_dept",
        )
        session.add_all([hr_user, viewer_user])
        await session.commit()
        print("Тестові дані успішно додано!")


if __name__ == "__main__":
    asyncio.run(seed())
