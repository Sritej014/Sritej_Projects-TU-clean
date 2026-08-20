from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

import json
import os
import shutil
import io
import unittest
from unittest.mock import patch
import warnings


# ============================================================
# IGNORE WARNINGS
# ============================================================

warnings.filterwarnings("ignore")


# ============================================================
# FILE PATHS
# ============================================================

FILEPATH = "structured_restaurant_data.json"
BACKUP_PATH = "structured_restaurant_data.json.bak"


# ============================================================
# 3.1 DEFINE THE RESTAURANT SCHEMA
# ============================================================

class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)


# ============================================================
# EXAMPLE RESTAURANT
# ============================================================

EXAMPLE_RESTAURANT_PARAGRAPH = """
Down in Santa Monica, Mar de Cortez serves as a sun-drenched,
casual taqueria specializing in Baja-style seafood.

With a 4.2/5 rating, it captures the salt-air energy of the coast
through its signature beer-battered snapper tacos and zesty
octopus ceviche, making it a premier spot for open-air dining
near the pier.

Price range: $
"""


EXAMPLE_OUTPUT = """
{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier",
    "shortcomings": []
}
"""


# ============================================================
# PROMPT GENERATION
# ============================================================

def restaurant_data_structure_prompt_generation(restaurant_paragraph):

    base_system_msg = """
You are a restaurant-data extraction assistant.

Your task is to convert an unstructured restaurant description
into a well-defined JSON object suitable for indexing and search.

The JSON MUST follow exactly this structure:

{
    "name": "string",
    "location": "string",
    "type": "string",
    "food_style": "string",
    "rating": float or null,
    "price_range": integer or null,
    "signatures": ["string"],
    "vibe": "string or null",
    "environment": "string",
    "shortcomings": ["string"]
}

Rules:

1. Return ONLY valid JSON.
2. Do not include Markdown.
3. Do not include ```json code fences.
4. Do not include explanations.
5. Convert dollar signs into an integer:

   $    -> 1
   $$   -> 2
   $$$  -> 3
   $$$$ -> 4

6. If rating or price range is unavailable, use null.
7. If there are no shortcomings, use an empty list.
8. If there are no signature dishes, use an empty list.
"""

    base_user_prompt = f"""
Convert the following restaurant description into the required JSON schema.

Restaurant description:

{restaurant_paragraph}


Example input:

{EXAMPLE_RESTAURANT_PARAGRAPH}


Example output:

{EXAMPLE_OUTPUT}


Return ONLY the JSON object.
"""

    return base_system_msg, base_user_prompt


# ============================================================
# LLM MODEL
# ============================================================

def llm_model(system_msg, prompt_txt, params=None):

    model_id = "ibm/granite-4-h-small"

    project_id = "skills-network"

    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com"
    )

    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
        params=params
    )

    messages = [
        {
            "role": "system",
            "content": system_msg
        },
        {
            "role": "user",
            "content": prompt_txt
        }
    ]

    response = model.chat(
        messages=messages
    )

    output_text = response["choices"][0]["message"]["content"]

    return output_text


# ============================================================
# JSON REPAIR PROMPTS
# ============================================================

def JSON_auto_repair_prompts(candidate_json_output, error_message):

    auto_repair_system_msg = """
You are a JSON repair expert.

Your task is to correct invalid restaurant JSON so that it
conforms exactly to the required schema.

Return ONLY the corrected JSON object.

Do not include:
- explanations
- markdown
- ```json code fences
- comments
- text before or after the JSON
"""

    auto_repair_prompt = f"""
Correct the following invalid JSON.

Invalid JSON:

{candidate_json_output}


Validation error:

{error_message}


Required schema:

{{
    "name": "string",
    "location": "string",
    "type": "string",
    "food_style": "string",
    "rating": float or null,
    "price_range": integer or null,
    "signatures": ["string"],
    "vibe": "string or null",
    "environment": "string",
    "shortcomings": ["string"]
}}

Return ONLY valid JSON.
"""

    return auto_repair_system_msg, auto_repair_prompt


# ============================================================
# CLEAN POSSIBLE MARKDOWN FROM LLM
# ============================================================

def clean_llm_json(text):

    text = text.strip()

    # Remove ```json
    if text.startswith("```json"):
        text = text[7:]

    # Remove generic ```
    elif text.startswith("```"):
        text = text[3:]

    # Remove closing ```
    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


# ============================================================
# NEW DATA ENTRY PROCESS
# ============================================================

def new_data_entry_process(paragraph, itemId):
    system_msg, user_prompt = (
        restaurant_data_structure_prompt_generation(paragraph)
    )
    candidate_json_output = llm_model(
        system_msg,
        user_prompt
    )

    candidate_json_output = clean_llm_json(
        candidate_json_output
    )
    max_attempts = 3

    for attempt in range(max_attempts):

        try:

            print("\nValidating generated restaurant JSON...")
            restaurant = Restaurant.model_validate_json(
                candidate_json_output
            )

            print("✅ JSON validated successfully.")
            structured_restaurant = restaurant.model_dump()
            structured_restaurant["id"] = itemId
            return structured_restaurant

        except ValidationError as e:

            print("\n❌ VALIDATION ERROR:")
            print(e)
            print("\nINVALID OUTPUT:")
            print(candidate_json_output)
            if attempt == max_attempts - 1:
                raise ValueError(
                    "Unable to generate valid restaurant JSON "
                    "after multiple repair attempts."
                ) from e
            repair_system_msg, repair_prompt = (
                JSON_auto_repair_prompts(
                    candidate_json_output,
                    str(e)
                )
            )
            candidate_json_output = llm_model(
                repair_system_msg,
                repair_prompt
            )
            candidate_json_output = clean_llm_json(
                candidate_json_output
            )
            print("\n🔧 JSON repaired. Retrying validation...")


# ============================================================
# LOAD DATA
# ============================================================

def load_data(file_path):

    # File does not exist yet
    if not os.path.exists(file_path):
        return []

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        return data

    except json.JSONDecodeError:

        print("❌ Database JSON is corrupted.")

        return []

    except IOError as e:

        print(
            f"❌ Could not read database: {e}"
        )

        return []


# ============================================================
# SAVE DATA
# ============================================================

def save_data(file_path, backup_path, data):

    # --------------------------------------------------------
    # Create backup before modifying database
    # --------------------------------------------------------

    if os.path.exists(file_path):

        shutil.copy2(
            file_path,
            backup_path
        )

    # --------------------------------------------------------
    # Save database
    # --------------------------------------------------------

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# DISPLAY RESTAURANT
# ============================================================

def show_restaurant_card(res, index):

    print(
        f"\n{'=' * 50}"
    )

    print(
        f"Restaurant Record #{index}"
    )

    print(
        f"{'=' * 50}"
    )

    for key, value in res.items():

        print(
            f"{key}: {value}"
        )

    print(
        f"{'=' * 50}"
    )


# ============================================================
# HELPER FOR EDITING VALUES
# ============================================================

def convert_updated_value(key, new_value):

    # --------------------------------------------------------
    # Convert rating
    # --------------------------------------------------------

    if key == "rating":

        if new_value.lower() == "null":
            return None

        return float(new_value)

    # --------------------------------------------------------
    # Convert price range and ID
    # --------------------------------------------------------

    if key in ["price_range", "id"]:

        if new_value.lower() == "null":
            return None

        return int(new_value)

    # --------------------------------------------------------
    # Convert list fields
    # --------------------------------------------------------

    if key in [
        "signatures",
        "shortcomings"
    ]:

        if new_value.strip() == "":
            return []

        return [
            value.strip()
            for value in new_value.split(",")
        ]

    # --------------------------------------------------------
    # Nullable string
    # --------------------------------------------------------

    if key == "vibe":

        if new_value.lower() == "null":
            return None

    # Otherwise remain a string
    return new_value


# ============================================================
# RESTAURANT DATABASE UI
# ============================================================

def manage_restaurants(
    file_path=FILEPATH,
    backup_path=BACKUP_PATH
):

    while True:

        # Reload database each iteration
        data = load_data(file_path)

        print(
            f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}"
        )

        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input(
            "\nAction: "
        )

        # ====================================================
        # OPTION 1
        # BROWSE ALL RESTAURANTS
        # ====================================================

        if choice == "1":

            print(
                "\n--- Current Listings ---"
            )

            if len(data) == 0:

                print(
                    "No restaurants available."
                )

                continue

            for index, restaurant in enumerate(data):

                # If name does not exist -> N/A
                name = restaurant.get(
                    "name",
                    "N/A"
                )

                print(
                    f"{index}: {name}"
                )


        # ====================================================
        # OPTION 2
        # VIEW DETAILED RECORD
        # ====================================================

        elif choice == "2":

            try:

                index = int(
                    input(
                        "Enter record index: "
                    )
                )

                # Check valid index
                if 0 <= index < len(data):

                    res = data[index]

                    show_restaurant_card(
                        res,
                        index
                    )

                else:

                    print(
                        "invalid index."
                    )

            except (
                ValueError,
                TypeError
            ):

                print(
                    "invalid index."
                )


        # ====================================================
        # OPTIONS 3, 4, 5 ARE WRITE OPERATIONS
        # ====================================================

        elif choice in [
            "3",
            "4",
            "5"
        ]:

            # ------------------------------------------------
            # SECURITY WARNING
            # ------------------------------------------------

            print(
                "\n❗ SECURITY WARNING: "
                "You are entering write-mode."
            )

            print(
                "Changes will be saved "
                "to the database immediately."
            )

            confirm = input(
                "Are you sure? "
                "(type 'yes' to proceed): "
            ).lower()

            if confirm != "yes":

                print(
                    "Operation cancelled."
                )

                continue


            # =================================================
            # OPTION 3
            # ADD NEW RESTAURANT
            # =================================================

            if choice == "3":

                # Create an ID for the new restaurant
                itemId = (
                    1000000
                    + len(data)
                    + 1
                )

                # Ask user for restaurant paragraph
                user_para = input(
                    "Please enter a new restaurant description: "
                )

                try:

                    # -----------------------------------------
                    # Convert paragraph -> structured dictionary
                    # -----------------------------------------

                    output = new_data_entry_process(
                        user_para,
                        itemId
                    )

                    # -----------------------------------------
                    # Append to database LIST
                    # -----------------------------------------

                    data.append(
                        output
                    )

                    # -----------------------------------------
                    # Save updated database
                    # -----------------------------------------

                    save_data(
                        file_path,
                        backup_path,
                        data
                    )

                    print(
                        "✅ Restaurant added."
                    )

                except Exception as e:

                    print(
                        f"❌ Restaurant could not be added: {e}"
                    )


            # =================================================
            # OPTION 4
            # EDIT RESTAURANT
            # =================================================

            elif choice == "4":

                try:

                    edit_index = int(
                        input(
                            "Enter index for record "
                            "to be edited: "
                        )
                    )

                    # -----------------------------------------
                    # Validate index
                    # -----------------------------------------

                    if not (
                        0
                        <= edit_index
                        < len(data)
                    ):

                        print(
                            "invalid index."
                        )

                        continue

                    restaurant = data[
                        edit_index
                    ]

                    print(
                        "\nPress Enter without typing "
                        "anything to keep the current value."
                    )

                    # -----------------------------------------
                    # Iterate over each restaurant field
                    # -----------------------------------------

                    for key in list(
                        restaurant.keys()
                    ):

                        current_value = (
                            restaurant[key]
                        )

                        new_value = input(
                            f"{key} [{current_value}]: "
                        )

                        # Empty input = don't modify
                        if new_value.strip() == "":
                            continue

                        try:

                            converted_value = (
                                convert_updated_value(
                                    key,
                                    new_value
                                )
                            )

                            restaurant[key] = (
                                converted_value
                            )

                        except ValueError:

                            print(
                                f"⚠ Invalid value for '{key}'. "
                                "Keeping previous value."
                            )

                    # -----------------------------------------
                    # Save changes
                    # -----------------------------------------

                    save_data(
                        file_path,
                        backup_path,
                        data
                    )

                    print(
                        "✅ Record updated."
                    )

                except (
                    ValueError,
                    TypeError
                ):

                    print(
                        "invalid index."
                    )


            # =================================================
            # OPTION 5
            # DELETE RESTAURANT
            # =================================================

            elif choice == "5":

                try:

                    delete_index = int(
                        input(
                            "Enter index for record "
                            "to be deleted: "
                        )
                    )

                    # -----------------------------------------
                    # Validate index
                    # -----------------------------------------

                    if (
                        0
                        <= delete_index
                        < len(data)
                    ):

                        # Remove restaurant
                        data.pop(
                            delete_index
                        )

                        # Save updated database
                        save_data(
                            file_path,
                            backup_path,
                            data
                        )

                        print(
                            "✅ Restaurant deleted."
                        )

                    else:

                        print(
                            "invalid index."
                        )

                except (
                    ValueError,
                    TypeError
                ):

                    print(
                        "invalid index."
                    )


        # ====================================================
        # OPTION 6
        # EXIT
        # ====================================================

        elif choice == "6":

            print(
                "Exiting restaurant database."
            )

            break


        # ====================================================
        # INVALID MENU OPTION
        # ====================================================

        else:

            print(
                "Invalid input."
            )


# ============================================================
# UNIT TESTS
# ============================================================

class TestRestaurantDatabase(unittest.TestCase):

    def setUp(self):
        """
        Create a temporary clean database for testing.
        """

        self.test_file = (
            "structured_restaurant_data_unit_test.json"
        )

        self.test_file_backup = (
            "structured_restaurant_data_unit_test.json.bak"
        )

        self.initial_data = [
            {
                "name": "Test Cafe",
                "location": "Test City"
            }
        ]

        with open(
            self.test_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.initial_data,
                f
            )


    def tearDown(self):
        """
        Clean up test files after tests.
        """

        if os.path.exists(
            self.test_file
        ):

            os.remove(
                self.test_file
            )

        if os.path.exists(
            self.test_file_backup
        ):

            os.remove(
                self.test_file_backup
            )


    @patch("builtins.input")
    @patch(
        "sys.stdout",
        new_callable=io.StringIO
    )
    def test_add_and_delete_restaurant_success(
        self,
        mock_stdout,
        mock_input
    ):

        """
        Test Scenario:
        Add a new restaurant and then delete it.
        """

        mock_restaurant = """
The Copper Sprout is a high-concept, Modern Appalachian
farm-to-table destination that blends an industrial-chic
aesthetic with rustic forest charm, featuring reclaimed wood
and amber lighting to create a sophisticated yet cozy vibe.

Priced in the $$ category, the menu celebrates seasonal
foraging and local heritage, headlined by signature dishes
like Cast-Iron Smoked Trout with pickled fiddlehead ferns
and hand-foraged Wild Mushroom Risotto with aged goat cheese.

The experience is designed to be intimate and earthy,
making it a premier spot for those seeking high-quality,
smokehouse-influenced cuisine in a refined,
atmospheric setting.
"""

        # ----------------------------------------------------
        # ADD TEST
        # ----------------------------------------------------

        mock_input.side_effect = [
            "3",
            "yes",
            mock_restaurant,
            "6"
        ]

        try:

            manage_restaurants(
                self.test_file,
                self.test_file_backup
            )

        except SystemExit:

            pass

        # Read resulting database
        with open(
            self.test_file,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        print(data)

        # Should now contain two restaurants
        self.assertEqual(
            len(data),
            2
        )

        self.assertIn(
            "✅ Restaurant added.",
            mock_stdout.getvalue()
        )


        # ----------------------------------------------------
        # DELETE TEST
        # ----------------------------------------------------

        mock_input.side_effect = [
            "5",
            "yes",
            1,
            "6"
        ]

        try:

            manage_restaurants(
                self.test_file,
                self.test_file_backup
            )

        except SystemExit:

            pass

        with open(
            self.test_file,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        print(data)

        # Back to one restaurant
        self.assertEqual(
            len(data),
            1
        )


    @patch("builtins.input")
    @patch(
        "sys.stdout",
        new_callable=io.StringIO
    )
    def test_delete_security_cancel(
        self,
        mock_stdout,
        mock_input
    ):

        """
        Test Scenario:
        Try to delete but say "no" to security warning.
        """

        mock_input.side_effect = [
            "5",
            "no",
            "6"
        ]

        manage_restaurants(
            self.test_file,
            self.test_file_backup
        )

        with open(
            self.test_file,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        # Nothing should have been deleted
        self.assertEqual(
            len(data),
            1
        )

        self.assertIn(
            "Operation cancelled.",
            mock_stdout.getvalue()
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # OPTION 1:
    # Run the actual restaurant application
    # --------------------------------------------------------

    #manage_restaurants(
    #    FILEPATH,
    #    BACKUP_PATH
    #)


    # --------------------------------------------------------
    # OPTION 2:
    # To run the UNIT TESTS instead, COMMENT OUT
    # manage_restaurants() above and UNCOMMENT this:
    # --------------------------------------------------------

    unittest.main()