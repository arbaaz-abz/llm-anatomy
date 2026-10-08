// The kv-cache concept module: re-exports the lesson and mounts it with the track's data (template rule 1).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './kv-cache/content.js';

export { LESSON };
export default { slug: 'kv-cache', title: 'The KV cache', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
