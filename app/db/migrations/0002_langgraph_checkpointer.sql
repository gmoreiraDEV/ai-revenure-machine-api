-- LangGraph Postgres Checkpointer
-- Gerado a partir do schema real do banco (columns + constraints + indexes)

-- =========================
-- checkpoint_migrations
-- =========================
create table if not exists checkpoint_migrations (
    v integer not null,
    constraint checkpoint_migrations_pkey primary key (v)
);

-- =========================
-- checkpoints
-- =========================
create table if not exists checkpoints (
    thread_id text not null,
    checkpoint_ns text not null default ''::text,
    checkpoint_id text not null,
    parent_checkpoint_id text null,
    type text null,
    checkpoint jsonb not null,
    metadata jsonb not null default '{}'::jsonb,
    constraint checkpoints_pkey
        primary key (thread_id, checkpoint_ns, checkpoint_id)
);

create index if not exists checkpoints_thread_id_idx
    on checkpoints (thread_id);

-- =========================
-- checkpoint_blobs
-- =========================
create table if not exists checkpoint_blobs (
    thread_id text not null,
    checkpoint_ns text not null default ''::text,
    channel text not null,
    version text not null,
    type text not null,
    blob bytea null,
    constraint checkpoint_blobs_pkey
        primary key (thread_id, checkpoint_ns, channel, version)
);

create index if not exists checkpoint_blobs_thread_id_idx
    on checkpoint_blobs (thread_id);

-- =========================
-- checkpoint_writes
-- =========================
create table if not exists checkpoint_writes (
    thread_id text not null,
    checkpoint_ns text not null default ''::text,
    checkpoint_id text not null,
    task_id text not null,
    idx integer not null,
    channel text not null,
    type text null,
    blob bytea not null,
    task_path text not null default ''::text,
    constraint checkpoint_writes_pkey
        primary key (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

create index if not exists checkpoint_writes_thread_id_idx
    on checkpoint_writes (thread_id);
