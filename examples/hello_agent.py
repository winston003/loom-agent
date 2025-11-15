import asyncio
import loom
from loom.builtin.llms import MockLLM
from loom.builtin.memory import InMemoryMemory
 
async def main():
    # 使用内置的 MockLLM 创建 Agent - 无需 API 密钥
    agent = loom.agent(
        llm=MockLLM(responses=["Hello! I'm a Loom agent ready to help!"]),
        memory=InMemoryMemory()
    )
    
    result = await agent.ainvoke("Hi, who are you?")
    print(result)
 
if __name__ == "__main__":
    asyncio.run(main())