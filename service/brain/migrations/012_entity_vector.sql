-- Entity resolution candidates from vector neighbours (DESIGN.md write step 6: "the alias table and vector
-- neighbours"): an embedding of each entity's name, set when the entity is created. Small table, exact scan.
set search_path = brain, extensions, public;

alter table entity add column if not exists name_vec vector(1024);
