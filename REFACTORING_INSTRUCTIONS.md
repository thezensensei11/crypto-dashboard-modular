# Refactoring Instructions

Generated on: 2025-06-29 20:23:16

## Files Created (Add code to these)

- [ ] `.env.example`
- [ ] `Dockerfile`
- [ ] `README_NEW.md`
- [ ] `core/__init__.py`
- [ ] `core/config.py`
- [ ] `core/constants.py`
- [ ] `core/exceptions.py`
- [ ] `core/models.py`
- [ ] `data/infrastructure_adapter.py`
- [ ] `infrastructure/collectors/base.py`
- [ ] `infrastructure/collectors/rest.py`
- [ ] `infrastructure/collectors/websocket.py`
- [ ] `infrastructure/database/manager.py`
- [ ] `infrastructure/message_bus/bus.py`
- [ ] `infrastructure/scheduler/celery_app.py`
- [ ] `infrastructure/scheduler/tasks.py`
- [ ] `scripts/__init__.py`
- [ ] `scripts/init_infrastructure.py`
- [ ] `scripts/manual_backfill.py`
- [ ] `scripts/migrate_data.py`
- [ ] `scripts/monitor.sh`
- [ ] `scripts/run_collector.py`
- [ ] `scripts/run_processor.py`
- [ ] `scripts/test_infrastructure.py`
- [ ] `setup.sh`

## Files to Update

- [ ] `main.py` - Replace with new version
- [ ] Copy `.env.example` to `.env` and update settings

## Next Steps

1. Open each file listed above
2. Follow the TODO instructions in each file
3. Copy the corresponding code from the artifacts
4. Make scripts executable: `chmod +x setup.sh scripts/monitor.sh`
5. Run setup: `./setup.sh`
6. Test: `python scripts/test_infrastructure.py`
