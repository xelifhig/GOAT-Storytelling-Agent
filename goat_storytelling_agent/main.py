from storytelling_agent import StoryAgent
from datetime import datetime

backend_uri = 'https://t0qycop7o4vpum-5000.proxy.runpod.net'
writer = StoryAgent(backend_uri, form='book for children aged 3 to 5', backend="openai", max_tokens=4096, request_timeout=500)
novel_scenes = writer.generate_story('A small dwarf that wanted to learn how to fly with the help of the animals of the forest and some creative ideas')

# Convert list of scenes to a single string
story_text = '\n\n'.join(novel_scenes) if isinstance(novel_scenes, list) else str(novel_scenes)

# Generate timestamp and filename
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
filename = f'GeneratedStory_{timestamp}.txt'

# Save the story to file
with open(filename, 'w', encoding='utf-8') as f:
    f.write(story_text)

print(f"Story has been saved to: {filename}")
print("\nStory content:")
print(story_text)