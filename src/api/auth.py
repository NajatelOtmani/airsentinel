from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from src.config import settings

# Point tokenUrl to your registered route prefix
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def verify_credentials(username: str, password: str) -> bool:
    if username != settings.ADMIN_USERNAME:
        return False

    # Direct bcrypt verification without passlib wrapper
    try:
        password_bytes = password.encode("utf-8")
        hash_bytes = settings.ADMIN_PASSWORD_HASH.encode("utf-8")
        return bcrypt.checkpw(password_bytes, hash_bytes)
    except Exception as exc:
        print(f"Bcrypt verification failed: {exc}")
        return False


def create_access_token(username: str) -> str:
    # Convert minutes setting to integer explicitly
    expire_minutes = int(getattr(settings, "JWT_EXPIRE_MINUTES", 60))
    expire = datetime.now(timezone.utc) + timedelta(minutes=expire_minutes)

    payload = {"sub": username, "exp": expire}
    return jwt.encode(
        payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return username
    except JWTError:
        raise credentials_exception
