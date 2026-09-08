import asyncio

import asyncpg

from src.config import settings


async def main():
    conn = await asyncpg.connect(settings.database_url)

    tables = await conn.fetch(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
    )
    print("=== TABLES ===")
    for r in tables:
        print(r["table_name"])

    for tname in ["anomalies", "sensor_readings"]:
        print(f"\n=== COLUMNS: {tname} ===")
        cols = await conn.fetch(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name=$1",
            tname,
        )
        for c in cols:
            print(c["column_name"], c["data_type"])

    await conn.close()


asyncio.run(main())
