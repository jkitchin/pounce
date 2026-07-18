use pounce_observability::{
    IterCaptureGuard, ITER_TARGET, collector_scope, init_for_tests, with_iter_capture,
};

fn emit(iter: i64) {
    tracing::info!(target: ITER_TARGET, iter = iter, objective = 1.0);
}

#[test]
fn scoped_capture_composes_with_global_install() {
    init_for_tests();

    let ((), records) = with_iter_capture(|| emit(3));
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].iter, 3);

    let _scope = collector_scope();
    let guard = IterCaptureGuard::start();
    emit(4);
    let records = guard.finish();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].iter, 4);
}
