from __future__ import annotations

from sqlalchemy.orm import Session

from src.db.models import AuditLog


class AuditLogRepository:
    def create_audit_log(
        self,
        session: Session,
        *,
        endpoint: str,
        loan_id: str,
        actor_username: str | None,
        actor_role: str | None,
        status_code: int,
        latency_ms: float,
        risk_score: float | None = None,
        decision: str | None = None,
        error_message: str | None = None,
    ) -> AuditLog:
        row = AuditLog(
            endpoint=endpoint,
            loan_id=loan_id,
            actor_username=actor_username,
            actor_role=actor_role,
            status_code=status_code,
            latency_ms=latency_ms,
            risk_score=risk_score,
            decision=decision,
            error_message=error_message,
        )
        session.add(row)
        session.flush()
        return row
