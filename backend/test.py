import random
from faker import Faker
from werkzeug.security import generate_password_hash
from app import create_app, db
from app.models import User, UserRole

fake = Faker()

def generate_indonesian_phone():
    return "628" + "".join(random.choices("0123456789", k=9))

def generate_user_id():
    return "12345" + "".join(random.choices("0123456789", k=5))

def seed_custom_users(num=50):
    funny_javanese_names = [
        "Triman Santoso", "Wagini Rahayu", "Sutrisno Mulya", "Parjo Lestari", "Karti Wibowo",
        "Mulyono Wasis", "Sarinem Ayu", "Wagiyo Setya", "Painem Sekar", "Bernardo Yuyiya",
        "Kasinem Sri", "Paidi Waras", "Ximon Yeyayi", "Sutarmi Cahya", "Sumarno Arif",
        "Juminten Lintang", "Sarwono Bagus", "Paiman Surya"
    ]
    
    football_players = [
        "Lionel Messi", "Cristiano Ronaldo", "Erling Haaland", "Neymar Jr", "Kylian Mbappé",
        "Mohamed Salah", "Karim Benzema", "Luka Modric", "Kevin De Bruyne", "Zlatan Ibrahimovic",
        "Harry Kane", "Luis Suárez", "Robert Lewandowski", "Sadio Mané", "Antoine Griezmann"
    ]
    
    wwe_wrestlers = [
        "John Cena", "The Undertaker", "Roman Reigns", "The Rock", "Rey Mysterio",
        "Brock Lesnar", "Triple H", "Kurt Angle", "Shawn Michaels", "Edge",
        "Randy Orton", "CM Punk", "Batista", "Seth Rollins", "AJ Styles"
    ]
    
    naruto_characters = [
        "Naruto Uzumaki", "Sasuke Uchiha", "Kakashi Hatake", "Itachi Uchiha", "Hinata Hyuga",
        "Sakura Haruno", "Shikamaru Nara", "Rock Lee", "Might Guy", "Gaara",
        "Tsunade Senju", "Jiraiya", "Madara Uchiha", "Obito Uchiha", "Minato Namikaze"
    ]

    all_names = funny_javanese_names + football_players + wwe_wrestlers + naruto_characters
    used_names = random.sample(all_names, k=num) if num <= len(all_names) else random.choices(all_names, k=num)

    for name in used_names:
        user = User(
            id=generate_user_id(),
            full_name=name,
            user_name=fake.user_name(),
            password=generate_password_hash("User123!"),
            email=fake.email(),
            phone_number=generate_indonesian_phone(),
            role=UserRole.user,
            approved=random.choice([False, None]),
            device_id=None
        )
        db.session.add(user)

    db.session.commit()
    print(f"{num} themed users seeded successfully.")

app = create_app()

with app.app_context():
    seed_custom_users(50)
