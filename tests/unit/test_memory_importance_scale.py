"""记忆库 importance 量纲（0-10 整数）与 create_memory 字段对齐的回归测试。

对应 issue #262 / #264：
- MemoryRepository.create_memory 此前传入 Memory 模型上不存在的字段
  （tags/metadata/last_accessed_at/updated_at）导致恒失败且被静默吞掉；
- importance 列为 Integer，但仓储层按 0-1 浮点写入，清理与排序建立在
  混合量纲上而失效。
"""
import json
import sys
import time
from pathlib import Path

import pytest
from sqlalchemy import select

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PARENT = PACKAGE_ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from self_learning_EterU.config import PluginConfig
from self_learning_EterU.models.orm.memory import Memory
from self_learning_EterU.repositories.memory_repository import MemoryRepository
from self_learning_EterU.services.database.sqlalchemy_database_manager import (
    SQLAlchemyDatabaseManager,
)
from self_learning_EterU.services.state.enhanced_memory_graph_manager import (
    EnhancedMemoryGraphManager,
    MemoryGraph,
)


def _make_config(tmp_path):
    return PluginConfig(
        data_dir=str(tmp_path),
        db_type="sqlite",
        enable_web_interface=False,
    )


def _make_manager(config, db, graphs):
    manager = EnhancedMemoryGraphManager.__new__(EnhancedMemoryGraphManager)
    manager.config = config
    manager.db_manager = db
    manager.llm_adapter = None
    manager.memory_graphs = graphs
    return manager


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_memory_persists_full_record(tmp_path):
    config = _make_config(tmp_path)
    db = SQLAlchemyDatabaseManager(config)
    try:
        assert await db.start() is True
        async with db.get_session() as session:
            repo = MemoryRepository(session)
            record = await repo.create_memory(
                group_id="group-a",
                user_id="user-a",
                content="概念内容",
                memory_type="concept",
                importance=8,
                tags='["tag1"]',
                metadata='{"concept": "概念"}',
            )

            assert record is not None, "create_memory 应成功写入而不是静默失败"
            assert record.id is not None
            assert record.importance == 8
            assert record.tags == '["tag1"]'
            assert record.metadata_ == '{"concept": "概念"}'
            assert record.last_accessed > 0
            assert record.created_at > 0

            rows = (await session.execute(select(Memory))).scalars().all()
            assert len(rows) == 1
    finally:
        await db.stop()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_memory_clamps_importance_to_0_10(tmp_path):
    config = _make_config(tmp_path)
    db = SQLAlchemyDatabaseManager(config)
    try:
        assert await db.start() is True
        async with db.get_session() as session:
            repo = MemoryRepository(session)
            high = await repo.create_memory(
                group_id="group-a",
                user_id="user-a",
                content="过高",
                memory_type="concept",
                importance=99,
            )
            low = await repo.create_memory(
                group_id="group-a",
                user_id="user-a",
                content="过低",
                memory_type="concept",
                importance=-3,
            )
            assert high.importance == 10
            assert low.importance == 0
    finally:
        await db.stop()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clean_old_memories_deletes_only_old_low_importance_rows(tmp_path):
    config = _make_config(tmp_path)
    db = SQLAlchemyDatabaseManager(config)
    now = int(time.time())
    try:
        assert await db.start() is True
        async with db.get_session() as session:
            repo = MemoryRepository(session)
            for importance, age_days, tag in [
                (2, 40, "old-low"),
                (9, 40, "old-high"),
                (2, 1, "new-low"),
                (9, 1, "new-high"),
            ]:
                session.add(
                    Memory(
                        group_id="group-a",
                        user_id="user-a",
                        content=tag,
                        importance=importance,
                        memory_type="concept",
                        created_at=now - age_days * 24 * 3600,
                        last_accessed=now,
                        access_count=0,
                    )
                )
            await session.commit()

            deleted = await repo.clean_old_memories(
                group_id="group-a",
                days=30,
                importance_threshold=3,
            )

            assert deleted == 1, "只应删除超过保留天数且重要性低于阈值的记录"
            remaining = {
                row.content
                for row in (await session.execute(select(Memory))).scalars().all()
            }
            assert remaining == {"old-high", "new-low", "new-high"}
    finally:
        await db.stop()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_save_memory_graph_persists_and_updates_concepts(tmp_path):
    config = _make_config(tmp_path)
    db = SQLAlchemyDatabaseManager(config)
    now = time.time()

    graph = MemoryGraph()
    graph.G.add_node(
        "概念A",
        memory_items="内容A",
        weight=0.7,
        created_time=now,
        last_modified=now,
    )
    graph.G.add_node(
        "概念B",
        memory_items="内容B",
        weight=0.35,
        created_time=now,
        last_modified=now,
    )
    manager = _make_manager(config, db, {"group-a": graph})

    try:
        assert await db.start() is True
        await manager.save_memory_graph("group-a")

        async with db.get_session() as session:
            repo = MemoryRepository(session)
            rows = {
                row.metadata_ and json.loads(row.metadata_)["concept"]: row
                for row in (
                    await session.execute(
                        select(Memory).where(Memory.memory_type == "concept")
                    )
                )
                .scalars()
                .all()
            }
            assert set(rows) == {"概念A", "概念B"}
            assert rows["概念A"].importance == 7
            assert rows["概念B"].importance == 4

        # 再次保存应更新既有记录而不是重复插入
        graph.G.nodes["概念A"]["weight"] = 0.95
        graph.G.nodes["概念A"]["memory_items"] = "更新后的内容A"
        await manager.save_memory_graph("group-a")

        async with db.get_session() as session:
            rows = (
                await session.execute(
                    select(Memory).where(Memory.memory_type == "concept")
                )
            ).scalars().all()
            by_concept = {
                json.loads(row.metadata_)["concept"]: row for row in rows
            }
            assert len(rows) == 2, "重复保存不应产生重复行"
            assert by_concept["概念A"].importance == 10
            assert by_concept["概念A"].content == "更新后的内容A"
    finally:
        await db.stop()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_load_memory_graph_restores_weight_and_timestamps(tmp_path):
    config = _make_config(tmp_path)
    db = SQLAlchemyDatabaseManager(config)
    now = time.time()

    graph = MemoryGraph()
    graph.G.add_node(
        "概念A",
        memory_items="内容A",
        weight=0.7,
        created_time=now,
        last_modified=now,
    )
    manager = _make_manager(config, db, {"group-a": graph})

    try:
        assert await db.start() is True
        await manager.save_memory_graph("group-a")

        # 重启后重新加载：权重与时间戳应从持久化数据还原
        reloaded = _make_manager(config, db, {})
        await reloaded.load_memory_graph("group-a")

        node = reloaded.memory_graphs["group-a"].G.nodes["概念A"]
        assert abs(node["weight"] - 0.7) < 1e-9
        assert abs(node["created_time"] - now) < 1e-6

        # 重载后再保存不应把重要度覆盖成默认权重对应的 10
        await reloaded.save_memory_graph("group-a")
        async with db.get_session() as session:
            row = (
                await session.execute(
                    select(Memory).where(Memory.memory_type == "concept")
                )
            ).scalars().first()
            assert row is not None
            assert row.importance == 7
    finally:
        await db.stop()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cleanup_task_uses_config_threshold_scale(tmp_path):
    config = _make_config(tmp_path)
    db = SQLAlchemyDatabaseManager(config)
    now = int(time.time())

    graph = MemoryGraph()
    manager = _make_manager(config, db, {"group-a": graph})

    try:
        assert await db.start() is True
        async with db.get_session() as session:
            for importance, tag in [(2, "old-low"), (9, "old-high")]:
                session.add(
                    Memory(
                        group_id="group-a",
                        user_id="user-a",
                        content=tag,
                        importance=importance,
                        memory_type="concept",
                        created_at=now - 40 * 24 * 3600,
                        last_accessed=now,
                        access_count=0,
                    )
                )
            await session.commit()

        # 默认配置 memory_importance_threshold=0.3 → 内部阈值 3（0-10）
        await manager._cleanup_old_memories_task()

        async with db.get_session() as session:
            remaining = {
                row.content
                for row in (await session.execute(select(Memory))).scalars().all()
            }
        assert remaining == {"old-high"}, "0-1 阈值应换算到 0-10 量纲后清理低重要度旧记忆"
    finally:
        await db.stop()
