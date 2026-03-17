"""
Supabase client singleton.

Provides a configured Supabase client for database operations.
Uses service role key for backend operations (bypasses RLS when needed).
Uses anon key for auth operations (respects RLS).
"""

import os
from functools import lru_cache
from supabase import create_client, Client


@lru_cache()
def get_supabase_client() -> Client:
    """Returns a Supabase client using the service role key (full DB access)."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set. "
            "Get these from Supabase Dashboard > Project Settings > API."
        )

    return create_client(url, key)


@lru_cache()
def get_supabase_auth_client() -> Client:
    """Returns a Supabase client using the anon key (for auth flows, respects RLS)."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")

    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_ANON_KEY must be set. "
            "Get these from Supabase Dashboard > Project Settings > API."
        )

    return create_client(url, key)
