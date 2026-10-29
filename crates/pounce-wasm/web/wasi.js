// Minimal `wasi_snapshot_preview1` host for a compute-only module.
//
// POUNCE's wasm build imports 14 WASI functions. Only four of them do
// anything here — the solver writes its console output through `fd_write`,
// times itself with `clock_time_get`, seeds its PRNG from `random_get`, and
// reads (an empty) environment. The rest are filesystem entry points that
// exist because Rust's std links them in; the solver never reaches them on
// this path, so they return errno rather than pretending to work.
//
// Keeping the shim here (~60 lines) is what lets the demo run from a plain
// static server with no build step, no npm install, and no `wasm-bindgen`.

const ERRNO_SUCCESS = 0;
const ERRNO_BADF = 8;
const ERRNO_NOSYS = 52;

/**
 * @param {(text: string) => void} onStdout called with each chunk the module
 *   writes to fd 1 / fd 2.
 */
export function createWasi(onStdout) {
  let memory = null;
  const decoder = new TextDecoder();
  const dv = () => new DataView(memory.buffer);

  const wasi_snapshot_preview1 = {
    fd_write(fd, iovs, iovsLen, nwritten) {
      const view = dv();
      let written = 0;
      let text = '';
      for (let i = 0; i < iovsLen; i++) {
        const ptr = view.getUint32(iovs + i * 8, true);
        const len = view.getUint32(iovs + i * 8 + 4, true);
        if (len > 0) text += decoder.decode(new Uint8Array(memory.buffer, ptr, len));
        written += len;
      }
      view.setUint32(nwritten, written, true);
      if (text) onStdout(text);
      return ERRNO_SUCCESS;
    },

    // The solver times itself with differences of the monotonic clock
    // (`performance.now()`). The realtime clock (id 0) is the wall-clock
    // time, which a solve report stamps as its creation date.
    clock_time_get(id, _precision, out) {
      const ms = id === 0 ? performance.timeOrigin + performance.now() : performance.now();
      dv().setBigUint64(out, BigInt(Math.round(ms * 1e6)), true);
      return ERRNO_SUCCESS;
    },
    clock_res_get(_id, out) {
      dv().setBigUint64(out, 1000n, true);
      return ERRNO_SUCCESS;
    },

    random_get(buf, len) {
      crypto.getRandomValues(new Uint8Array(memory.buffer, buf, len));
      return ERRNO_SUCCESS;
    },

    // No environment, no args.
    environ_sizes_get(count, size) {
      const view = dv();
      view.setUint32(count, 0, true);
      view.setUint32(size, 0, true);
      return ERRNO_SUCCESS;
    },
    environ_get: () => ERRNO_SUCCESS,
    args_sizes_get(count, size) {
      const view = dv();
      view.setUint32(count, 0, true);
      view.setUint32(size, 0, true);
      return ERRNO_SUCCESS;
    },
    args_get: () => ERRNO_SUCCESS,

    proc_exit(code) {
      throw new Error(`wasm called proc_exit(${code})`);
    },
    sched_yield: () => ERRNO_SUCCESS,

    // No filesystem in the browser sandbox.
    fd_close: () => ERRNO_BADF,
    fd_fdstat_get: () => ERRNO_BADF,
    fd_fdstat_set_flags: () => ERRNO_BADF,
    fd_filestat_get: () => ERRNO_BADF,
    fd_prestat_get: () => ERRNO_BADF,
    fd_prestat_dir_name: () => ERRNO_BADF,
    fd_read: () => ERRNO_BADF,
    fd_readdir: () => ERRNO_BADF,
    fd_seek: () => ERRNO_BADF,
    path_create_directory: () => ERRNO_NOSYS,
    path_filestat_get: () => ERRNO_NOSYS,
    path_open: () => ERRNO_NOSYS,
  };

  return {
    imports: { wasi_snapshot_preview1 },
    /** Must be called with the instance's memory before any other export. */
    bind(instance) {
      memory = instance.exports.memory;
      // Reactor modules expose `_initialize`; call it when present.
      if (typeof instance.exports._initialize === 'function') {
        instance.exports._initialize();
      }
    },
  };
}
