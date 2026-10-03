import sqlite3


def create_database():

    connection = sqlite3.connect(
        "plantvision.db"
    )

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            image_name TEXT,

            plant TEXT,

            disease TEXT,

            status TEXT,

            confidence REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


if __name__ == "__main__":

    create_database()

    print(
        "PlantVision database created successfully!"
    )