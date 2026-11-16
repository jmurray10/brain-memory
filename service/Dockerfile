FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY brain ./brain
# /backups: BRAIN_BACKUP_DIR for the worker. A new named volume mounted there takes this owner, so the non-root
# user can write it. (Railway volumes mount as root: set RAILWAY_RUN_UID=0 on a service that writes one.)
RUN useradd --create-home --uid 10001 brain && mkdir /backups && chown -R brain /app /backups
USER brain
# Railway sets PORT. Migrations run on start (BRAIN_MIGRATE_ON_START=1). The worker service overrides the
# command with `python -m brain.worker --once` (REFERENCE.md).
CMD ["python", "-m", "brain.serve"]
