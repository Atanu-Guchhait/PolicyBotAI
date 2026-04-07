def build_chroma_filter(filters: dict):
    if not filters:
        return None

    conditions = [{k: v} for k, v in filters.items()]

    return conditions[0] if len(conditions) == 1 else {"$and": conditions}