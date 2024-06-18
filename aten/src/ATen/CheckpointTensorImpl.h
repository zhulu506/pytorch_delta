#pragma once  

#include <atomic>
#include <memory>
#include <numeric>
#include <random>
#include <queue>

#include <c10/core/Backend.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/CopyBytes.h>

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/Flags.h>
#include <c10/util/Logging.h>
#include <c10/util/python_stub.h>
#include <c10/core/TensorImpl.h>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
// #define TORCH_CHECK(a, ...) // profile mode

namespace at {

template<typename T>
struct EquivalentClassNode : intrusive_ptr_target {
  T t_unsafe;
  mutable intrusive_ptr<EquivalentClassNode> parent;

  explicit EquivalentClassNode(const T& t) : t_unsafe(t) { }

  bool is_root() {
    return !parent;
  }
  void release_resources() override {
    parent.reset();
  }
};

template<typename T>
intrusive_ptr<EquivalentClassNode<T>> find_root(const intrusive_ptr<EquivalentClassNode<T>>& n) {
  if (n->is_root()) {
    return n;
  } else {
    n->parent = find_root(n->parent);
    return n->parent;
  }
}

template<typename T>
T& get_t(const intrusive_ptr<EquivalentClassNode<T>>& n) {
  return find_root(n)->t_unsafe;
}

template<typename T>
static void update_t(const intrusive_ptr<EquivalentClassNode<T>>& n, const T& t) {
  find_root(n)->t_unsafe = t;
}

template<typename T>
intrusive_ptr<EquivalentClassNode<T>> merge(const std::function<T(const T&, const T&)>& merge_t,
                                            const intrusive_ptr<EquivalentClassNode<T>>& lhs,
                                            const intrusive_ptr<EquivalentClassNode<T>>& rhs) {
  auto l = find_root(lhs);
  auto r = find_root(rhs);
  if (l == r) {
    return l;
  }
  l->parent = r;
  r->t_unsafe = merge_t(l->t_unsafe, r->t_unsafe);
  return r;
}

template<typename T>
void static_release_resources(intrusive_ptr<T>& ptr) {
  ptr.reset();
}

template<typename T>
struct RefCell final : intrusive_ptr_target {
  mutable T value;

  RefCell(const T& t) : value(t) { }

  void release_resources() final {
    static_release_resources(value);
  }
};

using time_t = std::chrono::time_point<std::chrono::system_clock>;
using duration_t = std::chrono::system_clock::duration;

struct CheckpointInfo {
  duration_t compute_cost;
  double bandwidth = 0.35*32*1024*1024*1024/1000000;

  CheckpointInfo(duration_t compute_cost) :
    compute_cost(compute_cost) {
  }
  
  /*DELTA: Filter Function*/
  double cost(size_t memory, size_t staleness) const {
    TORCH_CHECK(memory > 0);
    TORCH_CHECK(staleness > 0);
    return 1 / static_cast<double>(memory * staleness);
  }
  /*DTR: Cost Function*/
  double compute_cost_func(size_t memory, size_t staleness) const {
    TORCH_CHECK(memory > 0);
    TORCH_CHECK(staleness > 0);
    return compute_cost.count() / static_cast<double>(memory * staleness);
  }
  /*DELTA: Swap Cost Function*/
  double swap_cost_(size_t memory) const {
    TORCH_CHECK(memory > 0);
    return static_cast<double>(memory) / bandwidth;
  }
  /*DELTA: Decision Function*/
  double fake_decision(size_t memory, size_t staleness) const {
    TORCH_CHECK(memory > 0);
    TORCH_CHECK(staleness > 0);
    double swap_cost = static_cast<double>(memory) / bandwidth;
    return (2 * swap_cost) / compute_cost.count();
  }
};

using Tensors = std::vector<Tensor>;
using rematerialize_function_t = std::function<Tensors(const Tensors&)>;
using mutate_function_t = std::function<void(const Tensors&)>;

class CheckpointTensorCell;

struct Unsafe { };

struct Rematerializer : intrusive_ptr_target {
  rematerialize_function_t func;
  std::vector<intrusive_ptr<CheckpointTensorCell>> inputs;
  std::vector<weak_intrusive_ptr<CheckpointTensorCell>> outputs;
  intrusive_ptr<EquivalentClassNode<CheckpointInfo>> ecn;
  
  /*DTR: Recompute Cost*/
  duration_t compute_cost;
  /*DELTA: Reload Cost*/
  duration_t swap_cost; 

  Rematerializer(const Unsafe&,
                 const rematerialize_function_t& func,
                 const std::vector<intrusive_ptr<CheckpointTensorCell>>& inputs,
                 duration_t compute_cost)  :
    func(func),
    inputs(inputs),
    compute_cost(compute_cost) {
  }

  void release_resources() final {
    func = rematerialize_function_t();
    inputs.clear();
    outputs.clear();
  }
  /*DTR: Recompute Function*/
  void remat();
  /*DELTA: Reload Function*/
  void reload();
  intrusive_ptr<EquivalentClassNode<CheckpointInfo>> get_ecn();
  CheckpointInfo get_cpi();
};

struct AliasPool : intrusive_ptr_target {
  std::vector<weak_intrusive_ptr<CheckpointTensorCell>> tensors;
  std::vector<weak_intrusive_ptr<CheckpointTensorCell>> neighbors;
  intrusive_ptr<Rematerializer> head_remat;
  intrusive_ptr<EquivalentClassNode<CheckpointInfo>> ecn;

  size_t lock_count = 0;
  size_t external_count = 0;

  /*DTR: Flag*/
  bool is_evicted = false;
  /*DELTA: Flag*/
  bool is_offloaded = false;
  bool onGPU = true;
  bool swapped = false;
  
  /*DTR*/
  size_t memory;
  time_t last_used_time;
  /*DELTA*/
  double swap_cost;

  /*DELTA: Offload and Reload*/
  std::shared_ptr<Storage> cpuStorage;
  // for prefetch 
  // std::queue<intrusive_ptr<CheckpointTensorCell>> offload_queue;
  // void prefetch();

  AliasPool(const Unsafe&, intrusive_ptr<Rematerializer> head_remat, size_t memory) :
    head_remat(head_remat),
    memory(memory),
    last_used_time(std::chrono::system_clock::now()) {
  }

  std::set<intrusive_ptr<EquivalentClassNode<CheckpointInfo>>> neighbor_ecn();

  /*DELTA: Filter Function*/
  double cost(time_t current_time);
  /*DTR: Cost Function*/
  double compute_cost_func(time_t current_time);
  /*DELTA: Decision Function*/
  double decision_func(time_t current_time);

  /*DTR: Evict Function*/
  void evict();
  /*DELTA: Offload Function*/
  void offload();

  /*DTR*/
  void set_not_evicted(const intrusive_ptr<AliasPool>& self);
  /*DELTA*/
  void set_not_offloaded(const intrusive_ptr<AliasPool>& self);

  bool evictable() const {
    return lock_count == 0 && head_remat;
  }
  void lock() {
    ++lock_count;
  }
  void unlock() {
    --lock_count;
  }
  void register_external() {
    ++external_count;
  }
  void release_external() {
    --external_count;
    if (external_count == 0) {
      if (lock_count > 0) {return;}
      TORCH_CHECK(lock_count == 0);
      if (memory > 0 && (!ecn) && head_remat) {
        evict();
      }
    }
  }
  void release_resources() final {
    tensors.clear();
    neighbors.clear();
    head_remat.reset();
  }
};

struct CheckpointTensorCell : intrusive_ptr_target {
  bool defined = false;
  bool is_undefined_tensor;

  /*DELTA*/
  bool evicted = false;
  bool offloaded = false;
  bool onGPU = true;
  double compute_cost;
  double swap_cost;

  std::unique_ptr<Tensor> t;

  DispatchKeySet key_set_;
  caffe2::TypeMeta dtype_;
  c10::optional<Device> optional_device_;
  
  /*DELTA*/
  std::shared_ptr<Storage> cpuStorage;
  Tensor cpuTensor;

  intrusive_ptr<AliasPool> pool;
  intrusive_ptr<Rematerializer> remat;

  explicit CheckpointTensorCell(const Tensor& t, const intrusive_ptr<AliasPool>& pool) : pool(pool) {
    fill(t);
  }
  explicit CheckpointTensorCell(const Tensor& t,
                                const intrusive_ptr<AliasPool>& pool,
                                const intrusive_ptr<Rematerializer>& remat) :
    pool(pool), remat(remat) {
    fill(t);
  }

  /*DTR: Evict Function*/
  void evict() {
    TORCH_CHECK(remat);

    /*DELTA*/
    onGPU = false;
    evicted = true;

    t.reset();
  }
  /*DELTA: Offload Function*/
  void offload() {
    TORCH_CHECK(remat);
    t.reset();
  }
  /*DELTA: Reload Function*/
  void reload();

  void fill(const Tensor& t);
  Tensor get() {
    if (! t) {
      /*DTR: Recompute*/
      if (evicted) {
        TORCH_CHECK(remat);
        remat->remat();
      } 
      /*DELTA: Offload*/
      else if (offloaded) {
        reload();
      } 
      else { 
        TORCH_CHECK(false); 
      }
    }
    TORCH_CHECK(t);
    TORCH_CHECK(! t->key_set().has(DispatchKey::Checkpoint));
    pool->last_used_time = std::chrono::system_clock::now();
    return *t;
  }
  void pin() {
    get();
    pool->head_remat.reset();
    remat.reset();
  }
  size_t memory() {
    TORCH_CHECK(defined);
    return pool->memory;
  }
  DispatchKeySet key_set() const {
    TORCH_CHECK(defined);
    return key_set_;
  }
  caffe2::TypeMeta dtype() const {
    TORCH_CHECK(defined);
    return dtype_;
  }
  c10::optional<Device> optional_device() const {
    TORCH_CHECK(defined);
    return optional_device_;
  }
  void release_resources() final {
    t.reset();
    pool.reset();
    remat.reset();
  }
};

size_t memory(const Tensor& t);

struct External : intrusive_ptr_target {
  intrusive_ptr<CheckpointTensorCell> value;

  External(const intrusive_ptr<CheckpointTensorCell>& value) : value(value) {
    value->pool->register_external();
  }
  External(const Tensor& value) :
    External(intrusive_ptr<CheckpointTensorCell>::make(value,
                          intrusive_ptr<AliasPool>::make(Unsafe(),
                                                         intrusive_ptr<Rematerializer>(),
                                                         memory(value)))) { }
  External(const Tensor& value,
           const intrusive_ptr<AliasPool>& pool,
           const intrusive_ptr<Rematerializer>& remat) :
    External(intrusive_ptr<CheckpointTensorCell>::make(value, pool, remat)) { }
  
  void release_resources() override;
};

inline DispatchKeySet convert_key_set(const DispatchKeySet& t) {
  if (!t.has(DispatchKey::Checkpoint)) {
    auto ret = t.add(DispatchKey::Checkpoint);
    return ret;
  }
  else {
    auto ret = t;
    return ret;
  }
}

struct CheckpointTensorImpl : TensorImpl {
  int id = gen_counter();
  static int counter;
  static int gen_counter() {
    return counter++;
  }
  std::string counter_name() const {
    return std::string("x") + std::to_string(id);
  }

  intrusive_ptr<RefCell<intrusive_ptr<External>>> ref;

  explicit CheckpointTensorImpl(const intrusive_ptr<RefCell<intrusive_ptr<External>>>& ref) :
    TensorImpl(convert_key_set(ref->value->value->key_set()),
               ref->value->value->dtype(),
               ref->value->value->optional_device()),
    ref(ref) {
    if (key_set().has(DispatchKey::Autograd)) {
      set_requires_grad(true);
    }
  }
  explicit CheckpointTensorImpl(const intrusive_ptr<External>& e) :
    CheckpointTensorImpl(intrusive_ptr<RefCell<intrusive_ptr<External>>>::make(e)) { }
  explicit CheckpointTensorImpl(const Tensor& t);

  void release_resources() final;
  static Tensors make(const std::string& name,
                      const rematerialize_function_t& remat,
                      const Tensors& inputs);
  static void mutate(const std::string& name,
                     const mutate_function_t& mutate,
                     const Tensors& inputs,
                     const std::vector<size_t>& mutate_idx);
  intrusive_ptr<TensorImpl> shallow_copy_and_detach(const VariableVersion& version_counter,
                                                    bool allow_tensor_metadata_change) const override;
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;

  int64_t dim() const override {
    return ref->value->value->get().dim();
  }
  int64_t numel() const override {
    return ref->value->value->get().numel();
  }
  IntArrayRef sizes() const override {
    return ref->value->value->get().sizes();
  }
  int64_t size(int64_t d) const override {
    return ref->value->value->get().size(d);
  }
  IntArrayRef strides() const override {
    return ref->value->value->get().strides();
  }
  int64_t stride(int64_t d) const override {
    return ref->value->value->get().stride(d);
  }
  bool has_storage() const override {
    return false;
  }
};

struct CheckpointPool {
  /*DELTA: Not Use sample_tensors*/
  bool sample_tensors = false;
  bool ignore_small_tensors = true;
  bool has_memory_budget = false;
  long memory_budget;
  /*DELTA: Prefetch_count*/
  int64_t prefetch_count = 5;
  
  std::vector<weak_intrusive_ptr<AliasPool>> aps;
  std::vector<weak_intrusive_ptr<External>> exts;
  std::random_device rd;
  std::mt19937 gen = std::mt19937(rd());
  /*DELTA: Offload_queue*/
  std::queue<intrusive_ptr<CheckpointTensorCell>> offload_queue;

  CheckpointPool() { }

  void add(const intrusive_ptr<AliasPool>&);
  void auto_evict();  // auto_evict() Call evict()
  void evict();
  /*DELTA: Prefetch Function*/
  void prefetch();
  // void clear_offload_queue(std::queue<intrusive_ptr<CheckpointTensorCell>> q);
};

struct MakeRawResult {
  std::vector<intrusive_ptr<External>> outputs;
  std::vector<int> aliases;
  duration_t time;
  intrusive_ptr<Rematerializer> rematerializer;
};

inline CheckpointTensorImpl* get_cpti(const Tensor& t) {
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  TORCH_CHECK(cpti != nullptr);
  return cpti;
}

inline intrusive_ptr<RefCell<intrusive_ptr<External>>> cell_from_tensor(const Tensor& t) {
  return get_cpti(t)->ref;
}

}