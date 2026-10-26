import { NavLink, Navigate, Route, Routes } from 'react-router-dom';
import JobPage from './pages/JobPage';
import JobsPage from './pages/JobsPage';
import NewJobPage from './pages/NewJobPage';

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          rodent<span>-act</span>
        </div>
        <nav>
          <NavLink to="/" end>
            新建分析
          </NavLink>
          <NavLink to="/jobs">任务列表</NavLink>
        </nav>
        <div className="spacer" />
        <div className="hint">悬尾实验 (TST) · 检测 → 追踪 → 0.5 s 窗口分类</div>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<NewJobPage />} />
          <Route path="/jobs" element={<JobsPage />} />
          <Route path="/jobs/:id" element={<JobPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
