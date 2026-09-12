# Motion2Scene pitch

**From executable motions to learnable decisions.**

Knowing how to crouch is not the same as knowing when to crouch. Motion2Scene
turns a humanoid’s executed motions into training scenes. We target obstacles
where different motions lead to different outcomes, verify the available
responses in simulation, and teach a sensor-based policy when to commit or keep
its options open. Instead of asking the robot to learn from arbitrary obstacles,
we build its lessons around what it can actually execute.

English: 71 words. This describes the implemented method, without claiming
completed held-out or acquisition-efficiency gains.

**从可执行的动作，到可学会的选择。**

会下蹲，不等于知道什么时候该蹲。Motion2Scene根据机器人在仿真中实际执行的动作，反过来设计训练场景，让不同选择产生不同结果。我们验证完整动作的通行与恢复，保留每次失败记录，再教机器人根据所见，决定现在切换，还是继续前进、保留后续选择。核心是把已有动作，变成教会正确决策的训练数据。

中文：145 个字符（含标点及方法名称），其中 120 个汉字。WAIT 继续执行中性动作，并保留后续合法切换机会。
