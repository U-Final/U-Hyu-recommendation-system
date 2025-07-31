import pandas as pd
from sqlalchemy import text

# def load_user_data(conn):
#     return pd.DataFrame(conn.execute(text(
#         "SELECT id, gender, age_range FROM users"
#     )).fetchall(), columns=["user_id", "gender", "age_range"])

def load_user_data(conn, user_ids=None):
    if user_ids:
        placeholder = ','.join([':id{}'.format(i) for i in range(len(user_ids))])
        params = {f'id{i}': uid for i, uid in enumerate(user_ids)}
        query = f"SELECT id, gender, age_range FROM users WHERE id IN ({placeholder})"
        result = conn.execute(text(query), params)
    else:
        result = conn.execute(text("SELECT id, gender, age_range FROM users"))

    return pd.DataFrame(result.fetchall(), columns=["user_id", "gender", "age_range"])

def load_brand_data(conn):
    return pd.DataFrame(conn.execute(text(
        "SELECT id, brand_name, category_id FROM brands"
    )).fetchall(), columns=["brand_id", "brand_name", "category_id"])

def load_user_brand_data(conn, user_ids=None):
    base_query = """
        SELECT user_id, brand_id, data_type FROM (
            SELECT user_id, brand_id, 'INTEREST' as data_type FROM recommendation_base_data
            UNION
            SELECT DISTINCT user_id, brand_id, 'RECENT' as data_type FROM history WHERE visited_at IS NOT NULL
        ) AS combined
    """
    if user_ids:
        placeholder = ','.join([f':id{i}' for i in range(len(user_ids))])
        base_query += f" WHERE user_id IN ({placeholder})"
        params = {f'id{i}': uid for i, uid in enumerate(user_ids)}
        result = conn.execute(text(base_query), params)
    else:
        result = conn.execute(text(base_query))
    return pd.DataFrame(result.fetchall(), columns=["user_id", "brand_id", "data_type"])

def load_interaction_data(conn, user_ids=None):
    base_query = """
        SELECT al.user_id, b.id AS brand_id, al.action_type
        FROM action_logs al
        JOIN store s ON al.store_id = s.id
        JOIN brands b ON s.brand_id = b.id
        WHERE al.action_type IN ('MARKER_CLICK', 'FILTER_USED')
    """
    if user_ids:
        placeholder = ','.join([f':id{i}' for i in range(len(user_ids))])
        base_query += f" AND al.user_id IN ({placeholder})"
        params = {f'id{i}': uid for i, uid in enumerate(user_ids)}
        result = conn.execute(text(base_query), params)
    else:
        result = conn.execute(text(base_query))

    interaction_raw = pd.DataFrame(result.fetchall(), columns=["user_id", "brand_id", "action_type"])
    action_weights = {"MARKER_CLICK": 0.5, "FILTER_USED": 0.3}
    interaction_raw["weight"] = interaction_raw["action_type"].map(action_weights)
    return interaction_raw.groupby(["user_id", "brand_id"])["weight"].sum().reset_index()

def load_bookmark_data(conn, user_ids=None):
    base_query = """
        SELECT bl.user_id, s.brand_id
        FROM bookmark b
        JOIN bookmark_list bl ON b.bookmark_list_id = bl.id
        JOIN store s ON b.store_id = s.id
    """
    if user_ids:
        placeholder = ','.join([f':id{i}' for i in range(len(user_ids))])
        base_query += f" WHERE bl.user_id IN ({placeholder})"
        params = {f'id{i}': uid for i, uid in enumerate(user_ids)}
        result = conn.execute(text(base_query), params)
    else:
        result = conn.execute(text(base_query))

    return pd.DataFrame(result.fetchall(), columns=["user_id", "brand_id"])