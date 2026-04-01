import os
import json
import numpy as np

# ---- Load character info from JSON ----

# Build path like: <project_folder>/data/characters.json
BASE_DIR = os.path.dirname(__file__)
CHAR_FILE = os.path.join(BASE_DIR, "data", "characters.json")

with open(CHAR_FILE, "r", encoding="utf-8") as f:
    CHAR_INFO = json.load(f)

train_texts = [
    # --- LUKE SKYWALKER (20) ---
    "farm boy from Tatooine who became a Jedi Knight",
    "blew up the first Death Star with the Force",
    "son of Anakin Skywalker and Padme Amidala",
    "trained by Jedi Master Yoda on Dagobah",
    "rescued Han Solo from Jabba the Hutt",
    "strong with the light side of the Force",
    "wields a blue lightsaber before constructing a green one",
    "grew up with Uncle Owen and Aunt Beru on a moisture farm",
    "helped redeem Darth Vader during the Battle of Endor",
    "became a Jedi Master and rebuilt the Jedi Order",
    "fought Darth Vader on Cloud City and lost his hand",
    "piloted an X-wing during the Rebellion",
    "known for courage, compassion, and hope",
    "helped bring balance to the Force",
    "confronted Emperor Palpatine aboard the second Death Star",
    "early member of the Rebel Alliance",
    "rescued Princess Leia with Han Solo",
    "defeated Jabba's forces during the Sarlacc battle",
    "searched for Jedi teachings after the fall of the Empire",
    "helped inspire the Resistance decades later",

    # --- DARTH VADER (20) ---
    "Sith Lord in black armor who serves Emperor Palpatine",
    "formerly Anakin Skywalker, a Jedi Knight",
    "wields a red lightsaber and uses the dark side",
    "wears life-supporting black armor and mask",
    "father of Luke Skywalker and Princess Leia",
    "turned to the dark side after being manipulated by Palpatine",
    "fought Obi-Wan Kenobi on Mustafar and was severely injured",
    "destroyed many Jedi during Order 66",
    "known for his deep voice and intimidating presence",
    "commanded the Imperial fleet during the Galactic Civil War",
    "fought Luke Skywalker aboard Cloud City",
    "patrolled the galaxy hunting down rebels",
    "believed the dark side offered power and control",
    "feared throughout the Empire",
    "redeemed himself by saving Luke from Palpatine",
    "skilled pilot from childhood",
    "once the Chosen One of Jedi prophecy",
    "constructed his own red lightsaber as a Sith",
    "ruthless enforcer of Imperial rule",
    "conflicted by memories of Padme and his former life",

    # --- LEIA ORGANA (PRINCESS LEIA) (20) ---
    "Princess of Alderaan and daughter of Bail Organa",
    "leader of the Rebel Alliance against the Empire",
    "twin sister of Luke Skywalker",
    "daughter of Anakin Skywalker and Padme",
    "known for bravery and sharp leadership",
    "became a general in the Resistance",
    "helped steal the Death Star plans",
    "diplomat raised on Alderaan",
    "trained briefly in the Force",
    "saved Han Solo from Jabba the Hutt",
    "spoke out against Imperial tyranny",
    "worked closely with Mon Mothma",
    "helped coordinate the attack on the second Death Star",
    "fell in love with Han Solo",
    "disguised herself as Boushh to rescue Han",
    "one of the most influential leaders in the galaxy",
    "became a key figure in rebuilding the New Republic",
    "wore the classic white gown during the early Rebellion",
    "lost her home planet Alderaan to the Death Star",
    "a symbol of hope for the entire galaxy",

    # --- BOBA FETT (20) ---
    "feared bounty hunter wearing Mandalorian armor",
    "son and clone of Jango Fett",
    "worked for Jabba the Hutt",
    "captured Han Solo for a bounty",
    "nearly died falling into the Sarlacc pit",
    "wields EE-3 carbine rifle",
    "known for his silent, ruthless efficiency",
    "tracked targets across the galaxy",
    "survived the Sarlacc and reclaimed his armor",
    "rode with Din Djarin during later battles",
    "inherited his father's armor",
    "often flies a ship called Slave I",
    "collected bounties for the Empire",
    "wore green and yellow Mandalorian gear",
    "fought in battles on Tatooine",
    "rarely speaks but is deadly accurate",
    "one of the most feared hunters in the Outer Rim",
    "known for his iconic T-shaped visor helmet",
    "became ruler of Jabba's former territory",
    "expert tracker and combatant",

    # --- OBI-WAN KENOBI (20) ---
    "Jedi Master who trained Anakin Skywalker",
    "mentor to Luke Skywalker on Tatooine",
    "known for saying Hello there",
    "fought Darth Vader on Mustafar",
    "served as a general during the Clone Wars",
    "friend and partner of Jedi Master Qui-Gon Jinn",
    "taught Anakin before he fell to the dark side",
    "went by the name Ben Kenobi in exile",
    "gave Luke his father's lightsaber",
    "fought Darth Maul on Naboo",
    "one of the wisest Jedi Masters",
    "served on the Jedi Council",
    "escaped Order 66 by going into hiding",
    "sensed the future of Luke and Leia",
    "known for calmness, patience, and wit",
    "helped rescue the Chancellor from General Grievous",
    "dueled General Grievous on Utapau",
    "guided Luke through the Force after death",
    "expert in Form III lightsaber combat",
    "critical figure in both the Republic and Rebellion",

    # --- Padme Amidala (20) ---
    "married to Anakin Skywalker",
    "was born the princess of Naboo",
    "became a senator in the galactic republic representing Naboo",
    "the mother of Luke Skywalker and Leia Skywalker",
    "fought in an arena on Geonosis",
    "nearly assassinated by Aurra Sing",
    "nearly assassinated by Zam Wessel",
    "used her hand maiden as a decoy when her life was in danger",
    "talked Boss Nass into a peace deal on naboo",
    "was good friends with Bail Organa",
    "she was a leading voice in the Loyalist Committee in the senate",
    "always fought for diplomacy and democracy",
    "she died of a broken heart",
    "her advisor was senator Sheev Palpatine",
    "was captured by General Grievous during the clone wars",
    "once had a relationship with Rush Clovis",
    "used a laser pistol as a weapon",
    "followed Anakin Skywalker to Mustafar",
    "advocated for the end of the war",
    "uncovered a plot inside the banking clans",

    # --- Ahsoka Tano (20) ---
    "left the Jedi Order",
    "was accused of bombing the Jedi Temple",
    "was good friends with Bariss Offee",
    "padawan of Anakin Skywalker",
    "blew up a droid factory on Geonosis",
    "good friends with Rex",
    "buried her clone battalion after order 66",
    "trained to survive order 66",
    "found Ezra Bridger",
    "used two lightsabers",
    "went to Mandalore to help with Maul",
    "defeated Maul in a duel on Mandalore",
    "worked with Din Jarrin to train Grogu",
    "dueled Darth Vader on Malachor",
    "set Maul free during order 66",
    "removed Rex's inhibitor chip",
    "assisted the Martez sisters with the pikes",
    "was captured by Trandoshans along with younglings",
    "refused to kill the clones during order 66",
    "uses white light sabers",
    
    # --- EMPEROR PALPATINE / DARTH SIDIOUS (20) ---
    "Dark Lord of the Sith who ruled the Galactic Empire",
    "secretly manipulated both sides of the Clone Wars",
    "known to the Sith as Darth Sidious",
    "mastermind behind the fall of the Jedi Order",
    "trained Darth Vader in the dark side of the Force",
    "former Supreme Chancellor of the Galactic Republic",
    "issued Order 66 to destroy the Jedi",
    "used political manipulation to gain absolute power",
    "wields powerful Sith lightning",
    "corrupted Anakin Skywalker to the dark side",
    "believed to be the most powerful Sith Lord",
    "deceived the Jedi for decades",
    "maintained two identities as Palpatine and Sidious",
    "ruled the galaxy through fear and control",
    "responsible for the creation of the Empire",
    "manipulated Count Dooku as his apprentice",
    "sat at the center of galactic politics",
    "used the dark side to cheat death",
    "final enemy confronted by Luke Skywalker",
    "ultimate architect of Sith domination",

    # --- HAN SOLO (20) ---
    "smuggler who became a hero of the Rebel Alliance",
    "captain of the Millennium Falcon",
    "best friends with Chewbacca",
    "shot first in a cantina on Tatooine",
    "rescued Princess Leia from the Death Star",
    "helped destroy the second Death Star",
    "frozen in carbonite by Darth Vader",
    "former spice smuggler turned general",
    "married Princess Leia Organa",
    "known for sarcasm and quick thinking",
    "owes Jabba the Hutt a lot of money",
    "flew the Millennium Falcon through hyperspace",
    "helped Luke Skywalker become a hero",
    "instrumental in the Battle of Endor",
    "smuggled goods across the galaxy",
    "lost the Falcon to Lando Calrissian",
    "won the Falcon back in a card game",
    "one of the Rebellion's bravest pilots",
    "shot down enemy TIE fighters",
    "symbol of loyalty and bravery",

    # --- DARTH MAUL (20) ---
    "Sith apprentice with a double-bladed red lightsaber",
    "trained by Darth Sidious in secret",
    "known for his red and black facial tattoos",
    "wields a dual lightsaber staff",
    "killed Jedi Master Qui-Gon Jinn",
    "fought Obi-Wan Kenobi on Naboo",
    "cut in half and fell down a reactor shaft",
    "survived through hatred and rage",
    "used mechanical spider legs to survive",
    "became leader of the Crimson Dawn",
    "sought revenge against Obi-Wan Kenobi",
    "one of the most feared Sith warriors",
    "master of aggressive lightsaber combat",
    "operated in the criminal underworld",
    "manipulated crime syndicates",
    "trained in the dark side from childhood",
    "fought both Jedi and Sith",
    "symbol of Sith brutality",
    "later ruled Mandalore",
    "saved from insanity by Mother Talzin"
]

train_labels = [
    # Luke (20)
    "Luke Skywalker","Luke Skywalker","Luke Skywalker","Luke Skywalker","Luke Skywalker",
    "Luke Skywalker","Luke Skywalker","Luke Skywalker","Luke Skywalker","Luke Skywalker",
    "Luke Skywalker","Luke Skywalker","Luke Skywalker","Luke Skywalker","Luke Skywalker",
    "Luke Skywalker","Luke Skywalker","Luke Skywalker","Luke Skywalker","Luke Skywalker",

    # Vader (20)
    "Darth Vader","Darth Vader","Darth Vader","Darth Vader","Darth Vader",
    "Darth Vader","Darth Vader","Darth Vader","Darth Vader","Darth Vader",
    "Darth Vader","Darth Vader","Darth Vader","Darth Vader","Darth Vader",
    "Darth Vader","Darth Vader","Darth Vader","Darth Vader","Darth Vader",

    # Leia (20)
    "Leia Organa (Princess Leia)","Leia Organa (Princess Leia)","Leia Organa (Princess Leia)","Leia Organa (Princess Leia)","Leia Organa (Princess Leia)",
    "Leia Organa (Princess Leia)","Leia Organa (Princess Leia)","Leia Organa (Princess Leia)","Leia Organa (Princess Leia)","Leia Organa (Princess Leia)",
    "Leia Organa (Princess Leia)","Leia Organa (Princess Leia)","Leia Organa (Princess Leia)","Leia Organa (Princess Leia)","Leia Organa (Princess Leia)",
    "Leia Organa (Princess Leia)","Leia Organa (Princess Leia)","Leia Organa (Princess Leia)","Leia Organa (Princess Leia)","Leia Organa (Princess Leia)",

    # Boba (20)
    "Boba Fett","Boba Fett","Boba Fett","Boba Fett","Boba Fett",
    "Boba Fett","Boba Fett","Boba Fett","Boba Fett","Boba Fett",
    "Boba Fett","Boba Fett","Boba Fett","Boba Fett","Boba Fett",
    "Boba Fett","Boba Fett","Boba Fett","Boba Fett","Boba Fett",

    # Obi-Wan (20)
    "Obi-Wan Kenobi","Obi-Wan Kenobi","Obi-Wan Kenobi","Obi-Wan Kenobi","Obi-Wan Kenobi",
    "Obi-Wan Kenobi","Obi-Wan Kenobi","Obi-Wan Kenobi","Obi-Wan Kenobi","Obi-Wan Kenobi",
    "Obi-Wan Kenobi","Obi-Wan Kenobi","Obi-Wan Kenobi","Obi-Wan Kenobi","Obi-Wan Kenobi",
    "Obi-Wan Kenobi","Obi-Wan Kenobi","Obi-Wan Kenobi","Obi-Wan Kenobi","Obi-Wan Kenobi",

    # Padme Amidala (20)
    "Padme Amidala","Padme Amidala","Padme Amidala","Padme Amidala","Padme Amidala",
    "Padme Amidala","Padme Amidala","Padme Amidala","Padme Amidala","Padme Amidala",
    "Padme Amidala","Padme Amidala","Padme Amidala","Padme Amidala","Padme Amidala",
    "Padme Amidala","Padme Amidala","Padme Amidala","Padme Amidala","Padme Amidala",

    # Ahsoka Tano (20)
    "Ahsoka Tano","Ahsoka Tano","Ahsoka Tano","Ahsoka Tano","Ahsoka Tano",
    "Ahsoka Tano","Ahsoka Tano","Ahsoka Tano","Ahsoka Tano","Ahsoka Tano",
    "Ahsoka Tano","Ahsoka Tano","Ahsoka Tano","Ahsoka Tano","Ahsoka Tano",
    "Ahsoka Tano","Ahsoka Tano","Ahsoka Tano","Ahsoka Tano","Ahsoka Tano",

    # Emperor Palpatine (Darth Sidious)
    "Emperor Palpatine (Darth Sidious)","Emperor Palpatine (Darth Sidious)",
    "Emperor Palpatine (Darth Sidious)","Emperor Palpatine (Darth Sidious)",
    "Emperor Palpatine (Darth Sidious)","Emperor Palpatine (Darth Sidious)",
    "Emperor Palpatine (Darth Sidious)","Emperor Palpatine (Darth Sidious)",
    "Emperor Palpatine (Darth Sidious)","Emperor Palpatine (Darth Sidious)",
    "Emperor Palpatine (Darth Sidious)","Emperor Palpatine (Darth Sidious)",
    "Emperor Palpatine (Darth Sidious)","Emperor Palpatine (Darth Sidious)",
    "Emperor Palpatine (Darth Sidious)","Emperor Palpatine (Darth Sidious)",
    "Emperor Palpatine (Darth Sidious)","Emperor Palpatine (Darth Sidious)",
    "Emperor Palpatine (Darth Sidious)","Emperor Palpatine (Darth Sidious)",

    # Han Solo
    "Han Solo","Han Solo","Han Solo","Han Solo","Han Solo",
    "Han Solo","Han Solo","Han Solo","Han Solo","Han Solo",
    "Han Solo","Han Solo","Han Solo","Han Solo","Han Solo",
    "Han Solo","Han Solo","Han Solo","Han Solo","Han Solo",

    # Darth Maul
    "Darth Maul","Darth Maul","Darth Maul","Darth Maul","Darth Maul",
    "Darth Maul","Darth Maul","Darth Maul","Darth Maul","Darth Maul",
    "Darth Maul","Darth Maul","Darth Maul","Darth Maul","Darth Maul",
    "Darth Maul","Darth Maul","Darth Maul","Darth Maul","Darth Maul"
]

# Mapping: Name ⇄ ID
unique_labels = sorted(set(train_labels))

label_to_id = {name: i for i, name in enumerate(unique_labels)}
id_to_label = {i: name for name, i in label_to_id.items()}

# Encode labels as integers
y = np.array([label_to_id[label] for label in train_labels], dtype = np.int32)

# Training texts
X = train_texts