// The attention lesson module (docs/storyboards/attention.md). The page itself lives in ./attention/.
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON } from './attention/content.js';

export { LESSON };
export default { slug: 'attention', title: 'Attention, step by step', mount: (el, ctx) => mountLesson(el, ctx, LESSON) };
