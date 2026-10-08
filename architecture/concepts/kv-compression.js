// The kv-compression concept module: re-exports the lesson and mounts it with the track's data (template rule 1).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './kv-compression/content.js';

export { LESSON };
export default { slug: 'kv-compression', title: 'MQA, GQA, MLA', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
