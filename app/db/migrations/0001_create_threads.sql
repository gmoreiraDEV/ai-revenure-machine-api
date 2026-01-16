create table if not exists threads (
    thread_id text not null,
    created_at timestamptz not null default now(),
    constraint threads_pkey primary key (thread_id)
);
