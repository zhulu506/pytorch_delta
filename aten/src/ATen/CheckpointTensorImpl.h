#pragma once

#include <atomic>
#include <memory>
#include <numeric>
#include <random>

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
#define TORCH_CHECK(a, ...) // profile mode

namespace at {

template<typename T>
struct EquivalentClassNode : intrusive_ptr_target {
  explicit EquivalentClassNode(const T& t) : t_unsafe(t) { }
  mutable intrusive_ptr<EquivalentClassNode> parent;
  bool is_root() {
    return !parent;
  }
  void release_resources() override {
    parent.reset();
  }
  T t_unsafe;
};

template<typename T>
T& get_t(const intrusive_ptr<EquivalentClassNode<T>>& n) {
  return find_root(n)->t_unsafe;
}

template<typename T>
static void update_t(const intrusive_ptr<EquivalentClassNode<T>>& n, const T& t) {
  find_root(n)->t_unsafe = t;
}

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

size_t memory(const Tensor& t);

template<typename T>
struct RefCell final : intrusive_ptr_target {
  mutable T value;
  void release_resources() final {
    static_release_resources(value);
  }
  RefCell(const T& t) : value(t) { }
};

template<typename T>
using Ref = intrusive_ptr<RefCell<T>>;

template<typename T>
void static_release_resources(intrusive_ptr<T>& ptr) {
  ptr.reset();
}

class CheckpointTensorCell;
using strong = intrusive_ptr<CheckpointTensorCell>;
using strongs = std::vector<strong>;
using weak = weak_intrusive_ptr<CheckpointTensorCell>;
using weaks = std::vector<weak>;
using Tensors = std::vector<Tensor>;
using rematerialize_function_t = std::function<Tensors(const Tensors&)>;
using mutate_function_t = std::function<void(const Tensors&)>;

using time_t = std::chrono::time_point<std::chrono::system_clock>;
using duration_t = std::chrono::system_clock::duration;

extern double param_recompute_times; // 声明 recompute_times 的超参数

struct CheckpointInfo {
  duration_t compute_cost;

  // 新增 recompute_times
  // double cost(size_t memory, size_t free_mem, size_t staleness, size_t recompute_times) const {
  double cost(size_t memory, size_t free_mem, size_t staleness) const {
  // double cost(size_t memory, size_t staleness) const {
    TORCH_CHECK(memory > 0);
    TORCH_CHECK(staleness > 0);

    // 原 dtr
    // double cost_value = compute_cost.count() / static_cast<double>(memory * staleness);

    // 新增 cost_value 变量存储计算结果
    // double cost_value = compute_cost.count() * std::pow(param_recompute_times, static_cast<double>(recompute_times)) / static_cast<double>((memory + free_mem) * staleness);
    
    // 新增 free_mem 变量
    double cost_value = compute_cost.count() / static_cast<double>((memory + free_mem) * staleness);

    std::cout << "memory: " << memory << std::endl;
    std::cout << "free_mem: " << free_mem << std::endl;
    std::cout << "staleness: " << staleness << std::endl;
    std::cout << "compute_cost: " << compute_cost.count() << std::endl;
    // std::cout << "recompute_times: " << recompute_times << std::endl;
    std::cout << "cost: " << cost_value << std::endl;

    return cost_value;
  }
  CheckpointInfo(duration_t compute_cost) :
    compute_cost(compute_cost) {
  }
};

using ecn_ptr = intrusive_ptr<EquivalentClassNode<CheckpointInfo>>;

struct Unsafe { };

struct Rematerializer : intrusive_ptr_target {
  rematerialize_function_t func;
  strongs inputs;
  weaks outputs;
  duration_t compute_cost;
  ecn_ptr ecn;
  Rematerializer(const Unsafe&,
                 const rematerialize_function_t& func,
                 const strongs& inputs,
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
  void remat();
  ecn_ptr get_ecn();
  CheckpointInfo get_cpi();
};

struct AliasPool : intrusive_ptr_target {
  weaks tensors;
  weaks neighbors;
  std::set<ecn_ptr> neighbor_ecn();
  size_t lock_count = 0;
  size_t external_count = 0;
  void lock() {
    ++lock_count;
  }
  void unlock() {
    --lock_count;
  }
  intrusive_ptr<Rematerializer> head_remat;
  bool evictable() const {
    return lock_count == 0 && head_remat;
  }
  bool is_evicted = false;
  size_t memory;
  // 新增 free_mem
  size_t free_mem = 0;
  time_t last_used_time;

  // 新增 recompute_times
  size_t recompute_times = 0;
  // 新增 recompute_times++
  void add_recompute_times() {
    ++recompute_times;
  }

  AliasPool(const Unsafe&, intrusive_ptr<Rematerializer> head_remat, size_t memory) :
    head_remat(head_remat),
    memory(memory),
    last_used_time(std::chrono::system_clock::now()) {
  }
  
  ecn_ptr ecn;
  double cost(time_t current_time);
  void evict();
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
  
  void set_not_evicted(const intrusive_ptr<AliasPool>& self);
  void release_resources() final {
    tensors.clear();
    neighbors.clear();
    head_remat.reset();
  }
};

struct CheckpointTensorCell : intrusive_ptr_target {
  std::unique_ptr<Tensor> t;
  bool defined = false;
  bool is_undefined_tensor;
  DispatchKeySet key_set_;
  DispatchKeySet key_set() const {
    TORCH_CHECK(defined);
    return key_set_;
  }
  caffe2::TypeMeta dtype_;
  caffe2::TypeMeta dtype() const {
    TORCH_CHECK(defined);
    return dtype_;
  }
  c10::optional<Device> optional_device_;
  c10::optional<Device> optional_device() const {
    TORCH_CHECK(defined);
    return optional_device_;
  }

  intrusive_ptr<AliasPool> pool;
  intrusive_ptr<Rematerializer> remat;
  void evict() {
    TORCH_CHECK(remat);
    t.reset();
  }
  void fill(const Tensor& t);
  explicit CheckpointTensorCell(const Tensor& t, const intrusive_ptr<AliasPool>& pool) : pool(pool) {
    fill(t);
  }
  explicit CheckpointTensorCell(const Tensor& t,
                                const intrusive_ptr<AliasPool>& pool,
                                const intrusive_ptr<Rematerializer>& remat) :
    pool(pool), remat(remat) {
    fill(t);
  }
  size_t memory() {
    TORCH_CHECK(defined);
    return pool->memory;
  }
  Tensor get() {
    if (! t) {
      TORCH_CHECK(remat);
      remat->remat();
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
  void release_resources() final {
    t.reset();
    pool.reset();
    remat.reset();
  }
};

struct External : intrusive_ptr_target {
  External(const strong& value) : value(value) {
    value->pool->register_external();
  }
  External(const Tensor& value) :
    External(strong::make(value,
                          intrusive_ptr<AliasPool>::make(Unsafe(),
                                                         intrusive_ptr<Rematerializer>(),
                                                         memory(value)))) { }
  External(const Tensor& value,
           const intrusive_ptr<AliasPool>& pool,
           const intrusive_ptr<Rematerializer>& remat) :
    External(strong::make(value, pool, remat)) { }
  strong value;
  void release_resources() override;
};

inline DispatchKeySet convert_key_set(const DispatchKeySet& t) {
  CHECK(!t.has(DispatchKey::Checkpoint));
  auto ret = t.add(DispatchKey::Checkpoint);
  return ret;
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

  Ref<intrusive_ptr<External>> ref;

  void release_resources() final;

  explicit CheckpointTensorImpl(const Ref<intrusive_ptr<External>>& ref) :
    TensorImpl(convert_key_set(ref->value->value->key_set()),
               ref->value->value->dtype(),
               ref->value->value->optional_device()),
    ref(ref) {
    if (key_set().has(DispatchKey::Autograd)) {
      set_requires_grad(true);
    }
  }

  explicit CheckpointTensorImpl(const intrusive_ptr<External>& e) :
    CheckpointTensorImpl(Ref<intrusive_ptr<External>>::make(e)) { }

  explicit CheckpointTensorImpl(const Tensor& t);

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
  std::vector<weak_intrusive_ptr<AliasPool>> aps;
  std::vector<weak_intrusive_ptr<External>> exts;
  std::random_device rd;
  std::mt19937 gen = std::mt19937(rd());

  bool sample_tensors = false;

  bool ignore_small_tensors = true;
  bool has_memory_budget = false;
  long memory_budget;
  void evict();
  void auto_evict();
  void clear_checkpointpool();
  void add(const intrusive_ptr<AliasPool>&);
  CheckpointPool();
};

inline CheckpointTensorImpl* get_cpti(const Tensor& t) {
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  TORCH_CHECK(cpti != nullptr);
  return cpti;
}

inline Ref<intrusive_ptr<External>> cell_from_tensor(const Tensor& t) {
  return get_cpti(t)->ref;
}

}
