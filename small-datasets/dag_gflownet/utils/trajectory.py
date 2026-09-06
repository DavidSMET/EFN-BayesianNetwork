import numpy as np
import jax.numpy as jnp
import optax

from numpy.random import default_rng

TRAJECTORY = 8

class DAGTrajectory:
    def __init__(self):
        self

    def get_start(self, replay, batch_size, rng=None):
        if rng is None:
            rng = default_rng()
        ends = rng.choice(len(replay), size=batch_size * 200, replace=False)

        # 向前回溯找到链头 从链头开始按步长截取子链
        keep_seqs = []
        stop = False
        for itmp in ends:
            seq = [itmp]
            cnt = 1
            current = itmp
            while replay._prev[current] != -1:
                current = int(replay._prev[current])
                seq.append(current)
                cnt += 1
            seq.reverse()  # 反转得到从链头到链尾的顺序

            # 从这条完整轨迹中提取长度为 TRAJECTORY 的子序列
            M = cnt - TRAJECTORY + 1
            if M > 0:
                for i in range(M):
                    if len(keep_seqs) >= batch_size:
                        stop = True
                        break
                    keep_seqs.append(seq[i: i+TRAJECTORY])
            if stop:
                break
        return keep_seqs, TRAJECTORY              