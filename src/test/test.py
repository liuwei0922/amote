import time

class Signal:
    def __init__(self, id_name, features=None, parent=None):
        self.id = id_name
        self.features = features or set() 
        self.parent = parent

    def __repr__(self):
        return f"[{self.id}]"


class MemoryDB:
    def __init__(self):
        self.records = []

    def save(self, raw_features, concept_sig, action_sig, reward):
        self.records.append({
            "raw": raw_features, "concept": concept_sig, 
            "action": action_sig, "reward": reward
        })

    def find_successful_memory(self, concept_sig):
        for mem in reversed(self.records):
            if mem["concept"].id == concept_sig.id and mem["reward"] > 0:
                return mem
        return None


class InputProcessor:
    def __init__(self):
        self.buffer = []

    def receive_environment(self, text_input):
        print(f"\n[外部世界] 发生事件: {text_input}")
        features = set(text_input.replace(",", "").split())
        
        raw_signal = Signal("RAW_SENSORY", features)
        self.buffer.append(raw_signal)
        return self.buffer.pop(0)

class OutputProcessor:
    def __init__(self):
        self.buffer = []
        
    def execute(self, action_signal):
        self.buffer.append(action_signal)
        action = self.buffer.pop(0)
        
        if action.id == "ACT_PET":
            print("[物理输出] 动作执行: 伸手去摸...")
        elif action.id == "ACT_RUN":
            print("[物理输出] 动作执行: 赶紧逃跑...")
        else:
            print(f"[物理输出] 未知动作: {action.id}")

class CoreProcessor:
    def __init__(self, memory_db, output_processor):
        self.stack = []
        self.max_stack_size = 3
        self.memory = memory_db
        self.out_proc = output_processor
        
        self.sig_furbeast = Signal("CONCEPT_FURBEAST", {"毛", "四脚"})
        self.concept_rules = {
            self.sig_furbeast: Signal("ACT_PET")
        }

    def compute_similarity(self, raw_features, concept_features):
        intersection = raw_features.intersection(concept_features)
        if len(concept_features) == 0: return 0
        return len(intersection) / len(concept_features)

    def process(self, raw_signal, reward_feedback=None):
        self.stack.append(raw_signal)
        if len(self.stack) > self.max_stack_size:
            self.stack.pop(0)

        print(f"  [核心栈] 当前栈内容: {self.stack}")
        current_raw = self.stack[-1]

        best_match = None
        best_action = None
        highest_sim = 0.0

        for concept_sig, action_sig in self.concept_rules.items():
            sim = self.compute_similarity(current_raw.features, concept_sig.features)
            if sim > 0.8:
                if best_match is None or sim > highest_sim or (sim == highest_sim and len(concept_sig.features) > len(best_match.features)):
                    highest_sim = sim
                    best_match = concept_sig
                    best_action = action_sig


        if not best_match:
            print("  [核心层] 无法归约，感到困惑。")
            return

        print(f"  [核心层] 归约成功! 原始信号 -> {best_match} (相似度: {highest_sim:.2f})")
        
        self.out_proc.execute(best_action)

        if reward_feedback is not None:
            print(f"  [环境反馈] Reward: {reward_feedback}")
            self.memory.save(current_raw.features, best_match, best_action, reward_feedback)

            if reward_feedback < 0:
                print("  [核心层] 警告：因果冲突！触发记忆检索与【概念裂变】...")
                self._concept_fission(current_raw.features, best_match)

    def _concept_fission(self, current_bad_features, concept_sig):
        good_memory = self.memory.find_successful_memory(concept_sig)
        
        if good_memory:
            good_features = good_memory["raw"]
            
            diff_bad = current_bad_features - good_features
            diff_good = good_features - current_bad_features
            
            print(f"    -> 对比历史: 发现差异特征! 坏情况多出 {diff_bad}, 好情况多出 {diff_good}")
            new_bad_sig = Signal(f"CONCEPT_DOG_LIKE", concept_sig.features.union(diff_bad), parent=concept_sig)
            new_good_sig = Signal(f"CONCEPT_CAT_LIKE", concept_sig.features.union(diff_good), parent=concept_sig)

            self.concept_rules[new_bad_sig] = Signal("ACT_RUN") 
            self.concept_rules[new_good_sig] = Signal("ACT_PET") 
            
            print(f"    -> 裂变完成! 生成新内部信号: {new_bad_sig} 和 {new_good_sig}")



if __name__ == "__main__":
    memory = MemoryDB()
    out_p = OutputProcessor()
    core_p = CoreProcessor(memory, out_p)
    in_p = InputProcessor()

    print("============ 实验第一天：遇到猫 ============")
    sig1 = in_p.receive_environment("毛 四脚 喵喵")
    core_p.process(sig1, reward_feedback=1)

    time.sleep(1)
    print("\n============ 实验第二天：遇到狗 ============")
    sig2 = in_p.receive_environment("毛 四脚 汪汪")
    core_p.process(sig2, reward_feedback=-1)

    time.sleep(1)
    print("\n============ 实验第三天：再次遇到狗 =========")
    sig3 = in_p.receive_environment("毛 四脚 汪汪")
    core_p.process(sig3, reward_feedback=1)