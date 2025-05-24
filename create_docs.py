import asyncio
import random
import copy as copy

from sqlalchemy import text

from docs_db_connection import AsyncSessionLocal
from copy import deepcopy


products = [
    {
        "model": "Iphone 15 pro MAX",
        "name": "Чохол APPLE",
        "description": "Оригінальний силіконовий чохол. Простий але вишуканий доступний в трьох кольорах: прозорий, білий та чорний",
        "price": 890
    },
    {
        "model": "Iphone 13 pro MAX",
        "name": "Чохол APPLE",
        "description": "Оригінальний силіконовий чохол. Простий але вишуканий доступний в трьох кольорах: прозорий, білий та чорний",
        "price": 800
    },
    {
        "model": "Iphone 13 pro MAX",
        "name": "Чохол з MagSave",
        "description": "Силіконовий чохол MagSave. Доступний в наступних кольорах: рожевий, білий та чорний",
        "price": 950
    },
    {
        "model": "Iphone 15 pro",
        "name": "Чохол типу екзоскелет",
        "description": "Чохол типу екзоскелет, виготовлений з алюмінію. Доступний в наступних кольорах: нефарбований алюміній, графітовий",
        "price": 1900
    },
    {
        "model": "Iphone 15 pro MAX",
        "name": "Чохол типу екзоскелет",
        "description": "Чохол типу екзоскелет, виготовлений з алюмінію. Доступний в наступних кольорах: нефарбований алюміній, графітовий",
        "price": 1900
    }
]

def generate_items():

    base_list = copy.deepcopy(products)

    # Варіанти модифікацій
    models = ["Iphone 13", "Iphone 13 mini", "Iphone 14", "Iphone 14 pro", "Iphone 14 pro MAX", "Iphone 15", "Iphone 15 pro", "Iphone 15 pro MAX"]
    names = ["Чохол APPLE", "Чохол з MagSave", "Чохол типу екзоскелет", "Шкіряний чохол", "Прозорий чохол"]
    colors = ["прозорий", "білий", "чорний", "рожевий", "графітовий", "червоний", "синій", "зелений"]
    materials = ["силіконовий", "шкіряний", "алюмінієвий", "пластиковий", "карбоновий"]

    # Генерація
    def generate_description(material, color_options):
        return f"{material.capitalize()} чохол. Доступний в наступних кольорах: {', '.join(color_options)}"

    generated_items = []

    for i in range(100):
        base = random.choice(base_list)
        model = random.choice(models)
        name = random.choice(names)
        material = random.choice(materials)
        color_count = random.randint(2, 4)
        selected_colors = random.sample(colors, color_count)
        description = generate_description(material, selected_colors)
        price = random.choice([550, 660, 790, 850, 890, 920, 990, 1050, 1500])

        item = {
            "model": model,
            "name": name,
            "description": description,
            "price": price
        }
        generated_items.append(item)
    return generated_items

async def main():
    async with AsyncSessionLocal() as session:
        stmt = """
            CREATE TABLE IF NOT EXISTS products (
                id SERIAL PRIMARY KEY,
                model VARCHAR,
                name VARCHAR,
                description TEXT,
                price INTEGER
            )
        """
        try:
            await session.execute(text(stmt))
            await session.commit()
        except Exception as e:
            print("*****ERROR*****")
            print(e)

        for product in products:
            key_str = ""
            value_str = ""
            for key in product.keys():
                key_str = ", ".join(product.keys())
                value_str = ", ".join(f"'{str(value)}'" for value in product.values())

                query_str = f"INSERT INTO products ({key_str}) VALUES ({value_str})"
                print("-" * 25)
                print(query_str)
                try:
                    await session.execute(text(query_str))
                    await session.commit()
                except Exception as e:
                    print("*****ERROR*****")
                    print(e)

        generated_products = generate_items()
        for product in generated_products:
            key_str = ""
            value_str = ""
            for key in product.keys():
                key_str = ", ".join(product.keys())
                value_str = ", ".join(f"'{str(value)}'" for value in product.values())

                query_str = f"INSERT INTO products ({key_str}) VALUES ({value_str})"
                print("-" * 25)
                print(query_str)
                try:
                    await session.execute(text(query_str))
                    await session.commit()
                except Exception as e:
                    print("*****ERROR*****")
                    print(e)



if __name__ == "__main__":
    asyncio.run(main())
