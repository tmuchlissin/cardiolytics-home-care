import random
from faker import Faker
from werkzeug.security import generate_password_hash
from app import create_app, db
from app.models import User, UserRole

fake = Faker()

def generate_indonesian_phone():
    # Membuat nomor telepon dengan format "08" diikuti 9 digit acak
    return "628" + "".join(random.choices("0123456789", k=9))

def generate_user_id():
    # ID 10 digit yang selalu diawali dengan "12345" dan diikuti 5 digit acak
    random_digits = "".join(random.choices("0123456789", k=5))
    return "12345" + random_digits

def seed_random_users(num=7):
    for i in range(num):
        user = User(
            id=generate_user_id(),
            full_name=fake.name(),
            user_name=fake.user_name(),
            password=generate_password_hash("User123!"),
            email=fake.email(),
            phone_number=generate_indonesian_phone(),
            role=random.choice([UserRole.user]),
            approved=random.choice([False, None]),
            device_id=None
        )
        db.session.add(user)
    db.session.commit()
    print(f"{num} random users seeded successfully.")

app = create_app()

with app.app_context():
    seed_random_users(7)
