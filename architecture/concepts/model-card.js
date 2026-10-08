// The model-card concept module: re-exports the lesson and mounts it with the track's data (template rule 1).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './model-card/content.js';

export { LESSON };
export default { slug: 'model-card', title: 'Reading a model card', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
