-- DESIGN.md: the brain runs no generative model. Claims come from the user's own model with
-- the captured turn (TurnIn.claims) and wait as 'received' until the write path checks them (exact quote, value
-- in quote, Jev verify), which moves them to 'pending' (checked, awaiting build step 5) or 'dropped'.
set search_path = brain, extensions, public;

alter table claim_candidate drop constraint if exists claim_candidate_status_check;
alter table claim_candidate add constraint claim_candidate_status_check
    check (status in ('received', 'pending', 'dropped', 'placed'));
alter table claim_candidate alter column status set default 'received';
create index if not exists claim_candidate_received on claim_candidate (account_id, episode_id) where status = 'received';
comment on column claim_candidate.extractor is 'who extracted the claim: the API client that sent it with the turn';
